//go:build !integration

package task

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// transformProvider reuses the shared MockProvider (Hugging Face defaults for
// every task) and shadows only what the pipeline tests need: a fixed chat
// endpoint plus optional request/response transform hooks consumed by
// testCodec.
type transformProvider struct {
	testutils.MockProvider

	encodeFunc func(body []byte) ([]byte, http.Header, error)
	decodeFunc func(body []byte, ct string) ([]byte, error)
	onDecode   func(params hfproviders.DecodeParams)
}

func (p transformProvider) ChatEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{Path: "/test-endpoint"}, nil
}

// testCodec is a Codec[jsonInferenceReq, jsonInferenceResp].
// Encode returns request headers, Decode receives response
// headers and is itself responsible for content-type validation.
type testCodec struct {
	p transformProvider
}

func (c testCodec) Encode(
	params hfproviders.EncodeParams[jsonInferenceReq],
) (body []byte, headers http.Header, err error) {
	body, err = json.Marshal(params.Request)
	if err != nil {
		return nil, nil, err
	}
	if c.p.encodeFunc != nil {
		return c.p.encodeFunc(body)
	}

	return body, http.Header{
		"Content-Type": {"application/json"},
		"Accept":       {"application/json"},
	}, nil
}

func (c testCodec) Decode(params hfproviders.DecodeParams) (jsonInferenceResp, error) {
	if c.p.onDecode != nil {
		c.p.onDecode(params)
	}

	ct := params.Headers.Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		return jsonInferenceResp{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + ct,
			Err:     nil,
		}
	}

	respBody := params.Body
	if c.p.decodeFunc != nil {
		raw, err := c.p.decodeFunc(respBody, ct)
		if err != nil {
			return jsonInferenceResp{}, err
		}
		respBody = raw
	}

	var out jsonInferenceResp
	if err := json.Unmarshal(respBody, &out); err != nil {
		if errors.Is(err, io.EOF) {
			return jsonInferenceResp{}, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "empty response body",
				Err:     err,
			}
		}

		return jsonInferenceResp{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode response body",
			Err:     err,
		}
	}

	return out, nil
}

type jsonInferenceReq struct {
	Inputs string `json:"inputs"`
}

type jsonInferenceResp struct {
	GeneratedText string `json:"generated_text"`
}

func TestDoInference_Success(t *testing.T) {
	t.Parallel()

	respBody := `{"generated_text":"hello world"}`
	mt := testutils.NewJSONMockTransport(http.StatusOK, respBody, nil)
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(transformProvider{}),
	)
	p := transformProvider{}

	result, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, "hello world", result.GeneratedText)
	// Codec-supplied headers are applied to the request.
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Content-Type"))
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Accept"))
}

func TestDoInference_ProviderTransformsRequestAndResponse(t *testing.T) {
	t.Parallel()

	wrappedPrefix := `{"hfgo_inner":`
	wrappedSuffix := `}`

	mt := testutils.NewMockTransport(
		http.StatusOK,
		`{"hfgo_inner":{"generated_text":"hello"}}`,
		nil,
	)
	mt.Response.Header.Set("Content-Type", "application/json")

	encodeCalled := false
	decodeCalled := false

	p := transformProvider{
		encodeFunc: func(body []byte) ([]byte, http.Header, error) {
			encodeCalled = true

			return append(
				[]byte(wrappedPrefix),
				append(body, wrappedSuffix...)...,
			), http.Header{
				"Content-Type": {"application/json"},
				"Accept":       {"application/json"},
			}, nil
		},
		decodeFunc: func(body []byte, _ string) ([]byte, error) {
			decodeCalled = true
			if len(body) > len(wrappedPrefix)+len(wrappedSuffix) {
				inner := body[len(wrappedPrefix) : len(body)-len(wrappedSuffix)]

				return inner, nil
			}

			return body, nil
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	result, err := doInference(
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, "hello", result.GeneratedText)
	require.True(t, encodeCalled, "Encode should have been called")
	require.True(t, decodeCalled, "Decode should have been called")
}

func TestDoInference_EncodeErrorPropagated(t *testing.T) {
	t.Parallel()

	encodeErr := errors.New("encode failed")
	p := transformProvider{
		encodeFunc: func(_ []byte) ([]byte, http.Header, error) {
			return nil, nil, encodeErr
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(nil) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.ErrorIs(t, err, encodeErr)
}

func TestDoInference_DecodeErrorPropagated(t *testing.T) {
	t.Parallel()

	decodeErr := errors.New("decode failed")
	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hello"}`, nil)
	p := transformProvider{
		decodeFunc: func(_ []byte, _ string) ([]byte, error) {
			return nil, decodeErr
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.ErrorIs(t, err, decodeErr)
}

func TestDoInference_204NoContent(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusNoContent, "", nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	result, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, jsonInferenceResp{}, result)
}

func TestDoInference_205ResetContent(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusResetContent, "", nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	result, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, jsonInferenceResp{}, result)
}

func TestDoInference_EmptyResponseBody(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, "", nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoInference_InvalidJSONResponse(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `not json`, nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoInference_NonJSONResponseContentType(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	mt.Response.Header.Set("Content-Type", "text/plain")
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	// The pipeline no longer enforces response content types; the codec
	// surfaces the mismatch as a serialization error.
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoInference_UserHeadersOverrideCodecHeaders(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
		hfopts.WithHeader("Content-Type", "text/plain"),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	// Codec headers are defaults: the caller's explicit Content-Type wins
	// and the pipeline forwards the request as configured.
	require.NoError(t, err)
	require.Equal(t, "text/plain", mt.LastRequest.Header.Get("Content-Type"))
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Accept"))
}

func TestDoInference_CodecHeadersCannotOverrideAuth(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	p := transformProvider{
		encodeFunc: func(body []byte) ([]byte, http.Header, error) {
			return body, http.Header{
				"Content-Type":  {"application/json"},
				"Accept":        {"application/json"},
				"Authorization": {"Bearer codec-supplied"},
			}, nil
		},
	}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithToken("caller-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	// Authorization is caller-owned: codec-supplied values are dropped so
	// they can never replace the bearer token.
	require.NoError(t, err)
	require.Equal(t, "Bearer caller-token", mt.LastRequest.Header.Get("Authorization"))
}

func TestDoInference_MultiValuedCodecHeadersPreserved(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	p := transformProvider{
		encodeFunc: func(body []byte) ([]byte, http.Header, error) {
			return body, http.Header{
				"Content-Type": {"application/json"},
				"Accept":       {"application/json", "application/msgpack"},
			}, nil
		},
	}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(
		t,
		[]string{"application/json", "application/msgpack"},
		mt.LastRequest.Header.Values("Accept"),
	)
}

func TestDoStreamingInference_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"generated_text\":\"hello\"}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)

	// SSE framing is a transport concern: Accept is forced to
	// text/event-stream even though the codec defaults it to JSON.
	require.Equal(t, "text/event-stream", mt.LastRequest.Header.Get("Accept"))
}

func TestDoStreamingInference_ProviderTransformsPerEvent(t *testing.T) {
	t.Parallel()

	wrappedPrefix := `{"hfgo_inner":`
	wrappedSuffix := `}`

	body := "data: " + wrappedPrefix + `{"generated_text":"hello"}` + wrappedSuffix + "\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	p := transformProvider{
		decodeFunc: func(body []byte, _ string) ([]byte, error) {
			if len(body) > len(wrappedPrefix)+len(wrappedSuffix) {
				inner := body[len(wrappedPrefix) : len(body)-len(wrappedSuffix)]

				return inner, nil
			}

			return body, nil
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)
}

func TestDoStreamingInference_EventNamePropagated(t *testing.T) {
	t.Parallel()

	body := "event: chunk\ndata: {\"generated_text\":\"hello\"}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	var gotEvent string
	var gotHeaders http.Header

	p := transformProvider{
		onDecode: func(params hfproviders.DecodeParams) {
			gotEvent = params.Event
			gotHeaders = params.Headers
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)

	// The SSE event name and the synthesized JSON media type reach codecs
	// through DecodeParams.
	require.Equal(t, "chunk", gotEvent)
	require.Equal(t, "application/json", gotHeaders.Get("Content-Type"))
}

// framingCodec simulates an Anthropic-style provider: meaningful state
// travels in the SSE event name, frames like content_block_start carry no
// consumer-visible payload, and the stream ends on message_stop rather than
// a transport-level [DONE].
type framingCodec struct{}

func (framingCodec) Encode(
	_ hfproviders.EncodeParams[jsonInferenceReq],
) (body []byte, headers http.Header, err error) {
	return []byte(`{}`), http.Header{
		"Content-Type": {"application/json"},
		"Accept":       {"text/event-stream"},
	}, nil
}

func (framingCodec) Decode(params hfproviders.DecodeParams) (jsonInferenceResp, error) {
	switch params.Event {
	case "content_block_start":
		return jsonInferenceResp{}, hferrors.SkipEventError{}
	case "message_stop":
		return jsonInferenceResp{}, hferrors.EndOfStreamError{}
	default:
		var out jsonInferenceResp
		err := json.Unmarshal(params.Body, &out)

		return out, err
	}
}

func TestDoStreamingInference_CodecControlSignalsSkipAndEnd(t *testing.T) {
	t.Parallel()

	body := "event: content_block_start\ndata: {\"generated_text\":\"phantom\"}\n\n"
	body += "event: delta\ndata: {\"generated_text\":\"hello\"}\n\n"
	body += "event: message_stop\ndata: {}\n\n"
	body += "data: {\"generated_text\":\"unseen\"}\n\n"

	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(transformProvider{}),
	)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		framingCodec{},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	// content_block_start is suppressed; only the delta is delivered.
	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)

	// message_stop ends the stream; the trailing frame is never delivered.
	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
}

func TestDoStreamingInference_DecodeErrorPropagated(t *testing.T) {
	t.Parallel()

	decodeErr := errors.New("decode failed")
	body := "data: {\"generated_text\":\"hello\"}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	p := transformProvider{
		decodeFunc: func(_ []byte, _ string) ([]byte, error) {
			return nil, decodeErr
		},
	}

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, decodeErr)
}

func TestDoStreamingInference_NonEventStreamContentType(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	p := transformProvider{}
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(p),
	)

	_, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		hfproviders.Endpoint{Path: "/test-endpoint"},
		testCodec{p: p},
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}
