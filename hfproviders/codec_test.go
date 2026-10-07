//go:build !integration

package hfproviders

import (
	"context"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func jsonHeaders(ct string) http.Header {
	return http.Header{"Content-Type": {ct}}
}

// ctJSON is spelled to avoid a JSON-suffixed identifier so testifylint's
// encoded-compare heuristic treats header assertions as plain string
// comparisons.
const ctJSON = "application/json"

func assertSerializationError(t *testing.T, err error) {
	t.Helper()

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
}

type encodeReq struct {
	Inputs string `json:"inputs"`
}

type decodeResp struct {
	GeneratedText string `json:"generated_text"`
}

func TestJSONCodec_Encode(t *testing.T) {
	t.Parallel()

	body, headers, err := JSONCodec[encodeReq, decodeResp]{}.Encode(
		EncodeParams[encodeReq]{
			Context: context.Background(),
			Request: encodeReq{Inputs: "hi"},
			Model:   "some-model",
		},
	)
	require.NoError(t, err)
	require.JSONEq(t, `{"inputs":"hi"}`, string(body))
	require.Equal(t, ctJSON, headers.Get("Content-Type")) //nolint:testifylint // media type
	require.Equal(t, ctJSON, headers.Get("Accept"))       //nolint:testifylint // media type
}

func TestJSONCodec_EncodeError(t *testing.T) {
	t.Parallel()

	// Channels cannot be marshaled to JSON.
	_, _, err := JSONCodec[chan int, decodeResp]{}.Encode(
		EncodeParams[chan int]{Request: make(chan int)},
	)
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
	require.Error(t, sdkErr.Err, "underlying marshal error must be preserved")
}

func TestJSONCodec_Decode(t *testing.T) {
	t.Parallel()

	c := JSONCodec[encodeReq, decodeResp]{}

	t.Run("valid json", func(t *testing.T) {
		t.Parallel()

		resp, err := c.Decode(DecodeParams{
			Body:    []byte(`{"generated_text":"hello"}`),
			Headers: jsonHeaders(mimeTypeJSON),
		})
		require.NoError(t, err)
		require.Equal(t, "hello", resp.GeneratedText)
	})

	t.Run("json with charset parameter", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`{"generated_text":"hello"}`),
			Headers: jsonHeaders(mimeTypeJSON + "; charset=utf-8"),
		})
		require.NoError(t, err)
	})

	t.Run("structured plus json suffix", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`{"generated_text":"hello"}`),
			Headers: jsonHeaders("application/vnd.api+json"),
		})
		require.NoError(t, err)
	})

	t.Run("non-json content type", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`{"generated_text":"hello"}`),
			Headers: jsonHeaders("text/plain"),
		})
		require.Error(t, err)
		assertSerializationError(t, err)
	})

	t.Run("missing content type", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{Body: []byte(`{}`)})
		require.Error(t, err)
		assertSerializationError(t, err)
	})

	t.Run("empty body", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{Body: []byte{}, Headers: jsonHeaders(mimeTypeJSON)})
		require.Error(t, err)
		assertSerializationError(t, err)
		// The exact wrapping message depends on the Go version: some return
		// io.EOF (message "empty response body"), others a SyntaxError
		// ("unexpected end of JSON input"); assert only the invariant parts.
		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Error(t, sdkErr.Err, "underlying unmarshal error must be preserved")
	})

	t.Run("invalid json surfaces wrapped error", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`not json`),
			Headers: jsonHeaders(mimeTypeJSON),
		})
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Error(t, sdkErr.Err, "underlying unmarshal error must be preserved")
	})
}

func TestQuestionAnsweringCodec_Decode(t *testing.T) {
	t.Parallel()

	c := HuggingFaceCodecs{}.QuestionAnsweringCodec()
	params := func(body string) DecodeParams {
		return DecodeParams{
			Body:    []byte(body),
			Headers: jsonHeaders(mimeTypeJSON),
		}
	}

	t.Run("array response", func(t *testing.T) {
		t.Parallel()

		answers, err := c.Decode(params(
			`[{"answer":"Paris","score":0.9,"start":0,"end":5}]`,
		))
		require.NoError(t, err)
		require.Len(t, answers, 1)
		require.Equal(t, "Paris", answers[0].Answer)
	})

	t.Run("single object response", func(t *testing.T) {
		t.Parallel()

		answers, err := c.Decode(params(
			`{"answer":"Paris","score":0.9,"start":0,"end":5}`,
		))
		require.NoError(t, err)
		require.Len(t, answers, 1)
		require.Equal(t, "Paris", answers[0].Answer)
	})

	t.Run("structured plus json content type accepted", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`{"answer":"Paris","score":0.9,"start":0,"end":5}`),
			Headers: jsonHeaders("application/vnd.hf+json"),
		})
		require.NoError(t, err)
	})

	t.Run("non-json content type rejected", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`[{"answer":"Paris","score":0.9,"start":0,"end":5}]`),
			Headers: jsonHeaders("text/plain"),
		})
		require.Error(t, err)
		assertSerializationError(t, err)
		require.ErrorContains(t, err, "expected Content-Type application/json, got text/plain")
	})

	t.Run("unparseable response joins underlying errors", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(params(`not json`))
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Error(t, sdkErr.Err, "both unmarshal failures must be joined")
	})
}

func TestQuestionAnsweringCodec_EncodePassesHeaders(t *testing.T) {
	t.Parallel()

	body, headers, err := HuggingFaceCodecs{}.QuestionAnsweringCodec().Encode(
		EncodeParams[hftypes.QuestionAnsweringRequest]{
			Request: hftypes.QuestionAnsweringRequest{
				Input: hftypes.QuestionAnsweringInput{Question: "q", Context: "c"},
			},
		},
	)
	require.NoError(t, err)
	require.JSONEq(t, `{"inputs":{"question":"q","context":"c"}}`, string(body))
	//nolint:testifylint // media type, not a JSON document
	require.Equal(t, ctJSON, headers.Get("Content-Type"))
}

func TestTextClassificationCodec_Decode(t *testing.T) {
	t.Parallel()

	c := HuggingFaceCodecs{}.TextClassificationCodec()
	params := func(body string) DecodeParams {
		return DecodeParams{
			Body:    []byte(body),
			Headers: jsonHeaders(mimeTypeJSON),
		}
	}

	t.Run("flat response", func(t *testing.T) {
		t.Parallel()

		labels, err := c.Decode(params(`[{"label":"positive","score":0.95}]`))
		require.NoError(t, err)
		require.Len(t, labels, 1)
		require.Equal(t, "positive", labels[0].Label)
	})

	t.Run("single nested response unwraps outer layer", func(t *testing.T) {
		t.Parallel()

		labels, err := c.Decode(params(
			`[[{"label":"positive","score":0.95},{"label":"negative","score":0.05}]]`,
		))
		require.NoError(t, err)
		require.Len(t, labels, 2)
		require.Equal(t, "positive", labels[0].Label)
	})

	t.Run("multiple nested result sets rejected", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(params(
			`[[{"label":"a","score":0.5}],[{"label":"b","score":0.5}]]`,
		))
		require.Error(t, err)
		require.ErrorContains(t, err, "expected a single text-classification result set, got 2")
	})

	t.Run("non-json content type rejected", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(DecodeParams{
			Body:    []byte(`[{"label":"positive","score":0.95}]`),
			Headers: jsonHeaders("text/plain"),
		})
		require.Error(t, err)
		assertSerializationError(t, err)
		require.ErrorContains(t, err, "expected Content-Type application/json, got text/plain")
	})

	t.Run("unparseable response joins underlying errors", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(params(`not json`))
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Error(t, sdkErr.Err, "both unmarshal failures must be joined")
	})
}
