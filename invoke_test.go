//go:build !integration

package hfgo

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/Kardbord/hfgo/v4/providers"
	"github.com/stretchr/testify/require"
)

type transformProvider struct {
	providers.DefaultCodec

	encodeFunc func(task providers.Task, body []byte, ct string) ([]byte, string, error)
	decodeFunc func(task providers.Task, body []byte, ct string) ([]byte, string, error)
	suffix     string
}

func (p transformProvider) Endpoint(_ providers.Task, _ string) (string, error) {
	return "/test-endpoint", nil
}

func (p transformProvider) ProviderSuffix() string {
	return p.suffix
}

func (p transformProvider) EncodeRequest(
	task providers.Task, body []byte, ct string,
) (providerBody []byte, providerContentType string, err error) {
	if p.encodeFunc != nil {
		return p.encodeFunc(task, body, ct)
	}

	return body, ct, nil
}

func (p transformProvider) DecodeResponse(
	task providers.Task, body []byte, ct string,
) (hfBody []byte, hfContentType string, err error) {
	if p.decodeFunc != nil {
		return p.decodeFunc(task, body, ct)
	}

	return body, ct, nil
}

type jsonInferenceReq struct {
	Inputs string `json:"inputs"`
}

type jsonInferenceResp struct {
	GeneratedText string `json:"generated_text"`
}

func TestDoJSONInference_Success(t *testing.T) {
	t.Parallel()

	respBody := `{"generated_text":"hello world"}`
	mt := testutils.NewJSONMockTransport(http.StatusOK, respBody, nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	result, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, "hello world", result.GeneratedText)
}

func TestDoJSONInference_ProviderTransformsRequestAndResponse(t *testing.T) {
	t.Parallel()

	// The provider wraps the request body and unwraps the response body,
	// verifying that both EncodeRequest and DecodeResponse are applied.
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
		encodeFunc: func(_ providers.Task, body []byte, ct string) ([]byte, string, error) {
			encodeCalled = true

			return append([]byte(wrappedPrefix), append(body, wrappedSuffix...)...), ct, nil
		},
		decodeFunc: func(_ providers.Task, body []byte, ct string) ([]byte, string, error) {
			decodeCalled = true
			// Unwrap: strip {"hfgo_inner": prefix and } suffix
			if len(body) > len(wrappedPrefix)+len(wrappedSuffix) {
				inner := body[len(wrappedPrefix) : len(body)-len(wrappedSuffix)]

				return inner, ct, nil
			}

			return body, ct, nil
		},
	}

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(p)

	result, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, "hello", result.GeneratedText)
	require.True(t, encodeCalled, "EncodeRequest should have been called")
	require.True(t, decodeCalled, "DecodeResponse should have been called")
}

func TestDoJSONInference_EncodeErrorPropagated(t *testing.T) {
	t.Parallel()

	encodeErr := errors.New("encode failed")
	p := transformProvider{
		encodeFunc: func(_ providers.Task, _ []byte, ct string) ([]byte, string, error) {
			return nil, ct, encodeErr
		},
	}

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(nil) }).
		WithModel("test-model").
		WithProvider(p)

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.ErrorIs(t, err, encodeErr)
}

func TestDoJSONInference_DecodeErrorPropagated(t *testing.T) {
	t.Parallel()

	decodeErr := errors.New("decode failed")
	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hello"}`, nil)
	p := transformProvider{
		decodeFunc: func(_ providers.Task, _ []byte, ct string) ([]byte, string, error) {
			return nil, ct, decodeErr
		},
	}

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(p)

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.ErrorIs(t, err, decodeErr)
}

func TestDoJSONInference_204NoContent(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusNoContent, "", nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	result, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, jsonInferenceResp{}, result)
}

func TestDoJSONInference_205ResetContent(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusResetContent, "", nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	result, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	require.Equal(t, jsonInferenceResp{}, result)
}

func TestDoJSONInference_EmptyResponseBody(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, "", nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoJSONInference_InvalidJSONResponse(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `not json`, nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoJSONInference_NonJSONResponseContentType(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	mt.Response.Header.Set("Content-Type", "text/plain")
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestDoJSONInference_NoModel(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithProvider(transformProvider{})

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestDoJSONInference_NilProvider(t *testing.T) {
	t.Parallel()

	opts := request.NewOptions().
		WithModel("test-model").
		WithProvider(nil)

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestDoJSONInference_ContentTypeValidation(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, `{"generated_text":"hi"}`, nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{}).
		WithHeader("Content-Type", "text/plain")

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestDoJSONInference_ModelWithExistingSuffixPassedThrough(t *testing.T) {
	t.Parallel()

	respBody := `{"generated_text":"hello"}`
	p := suffixProvider{
		transformProvider: transformProvider{},
	}
	mt := testutils.NewJSONMockTransport(http.StatusOK, respBody, nil)
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("mistral-7b:sambanova").
		WithProvider(p)

	_, err := doJSONInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)

	require.NotNil(t, mt.LastRequest)
	// The model "mistral-7b" should have the suffix "sambanova" appended
	// by the service layer before reaching doJSONInference.
	require.Contains(t, mt.LastRequest.URL.Path, "mistral-7b:sambanova")
}

// suffixProvider is a transformProvider that includes the model in the endpoint path.
type suffixProvider struct {
	transformProvider
}

func (p suffixProvider) Endpoint(_ providers.Task, model string) (string, error) {
	return "/test-endpoint/" + model, nil
}

func TestDoStreamingInference_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"generated_text\":\"hello\"}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
}

func TestDoStreamingInference_ProviderTransformsPerEvent(t *testing.T) {
	t.Parallel()

	// The provider wraps each event's JSON data; the decode function unwraps it.
	wrappedPrefix := `{"hfgo_inner":`
	wrappedSuffix := `}`

	body := "data: " + wrappedPrefix + `{"generated_text":"hello"}` + wrappedSuffix + "\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	p := transformProvider{
		decodeFunc: func(_ providers.Task, body []byte, ct string) ([]byte, string, error) {
			if len(body) > len(wrappedPrefix)+len(wrappedSuffix) {
				inner := body[len(wrappedPrefix) : len(body)-len(wrappedSuffix)]

				return inner, ct, nil
			}

			return body, ct, nil
		},
	}

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(p)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.GeneratedText)
}

func TestDoStreamingInference_DecodeErrorPropagated(t *testing.T) {
	t.Parallel()

	decodeErr := errors.New("decode failed")
	body := "data: {\"generated_text\":\"hello\"}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	p := transformProvider{
		decodeFunc: func(_ providers.Task, _ []byte, ct string) ([]byte, string, error) {
			return nil, ct, decodeErr
		},
	}

	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(p)

	stream, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
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
	opts := request.NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }).
		WithModel("test-model").
		WithProvider(transformProvider{})

	_, err := doStreamingInference[jsonInferenceReq, jsonInferenceResp](
		opts,
		providers.TaskTextGeneration,
		jsonInferenceReq{Inputs: "hi"},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}
