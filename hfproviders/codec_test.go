//go:build !integration

package hfproviders

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func assertSerializationError(t *testing.T, err error) {
	t.Helper()

	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
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

type sdkErrorReq struct{}

func (sdkErrorReq) MarshalJSON() ([]byte, error) {
	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindConfiguration,
		Message: "request is invalid",
	}
}

type sdkErrorResp struct{}

func (*sdkErrorResp) UnmarshalJSON([]byte) error {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindValidation,
		Message: "response is invalid",
	}
}

func TestJSONCodec_EncodePreservesSDKError(t *testing.T) {
	t.Parallel()

	_, _, err := JSONCodec[sdkErrorReq, decodeResp]{}.Encode(
		EncodeParams[sdkErrorReq]{Request: sdkErrorReq{}},
	)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestJSONCodec_DecodePreservesSDKError(t *testing.T) {
	t.Parallel()

	_, err := JSONCodec[encodeReq, sdkErrorResp]{}.Decode(DecodeParams{
		Headers: jsonHeaders(ctJSON),
		Body:    []byte(`{}`),
	})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindValidation)
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

	c := HuggingFaceQuestionAnsweringMixin{}.QuestionAnsweringCodec()
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

	t.Run("empty object rejected", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(params(`{}`))
		require.Error(t, err)
		assertSerializationError(t, err)
	})

	t.Run("unrelated object rejected", func(t *testing.T) {
		t.Parallel()

		_, err := c.Decode(params(`{"error":"model failed"}`))
		require.Error(t, err)
		assertSerializationError(t, err)
	})
}

func TestExistingSDKError(t *testing.T) {
	t.Parallel()

	want := &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindValidation,
		Message: "invalid response",
		Err:     nil,
	}
	require.Same(t, want, existingSDKError(fmt.Errorf("wrapped: %w", want)))
	require.Same(
		t,
		want,
		existingSDKError(errors.Join(errors.New("first"), want)),
	)
	require.NoError(t, existingSDKError(errors.New("plain error")))
}

func TestDecodeSingleQuestionAnswering(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		body    string
		want    hftypes.QuestionAnswer
		wantErr string
	}{
		{
			name: "valid answer",
			body: `{"answer":"Paris","score":0.9,"start":0,"end":5}`,
			want: hftypes.QuestionAnswer{
				Answer: "Paris",
				Score:  0.9,
				Start:  0,
				End:    5,
			},
		},
		{
			name:    "missing answer",
			body:    `{"score":0.9}`,
			wantErr: "missing answer",
		},
		{
			name:    "missing score",
			body:    `{"answer":"Paris"}`,
			wantErr: "missing score",
		},
		{
			name:    "null answer",
			body:    `{"answer":null,"score":0.9}`,
			wantErr: "answer is null",
		},
		{
			name:    "null score",
			body:    `{"answer":"Paris","score":null}`,
			wantErr: "score is null",
		},
		{
			name:    "null answer and score",
			body:    `{"answer":null,"score":null}`,
			wantErr: "answer is null",
		},
		{
			name:    "invalid score type",
			body:    `{"answer":"Paris","score":"high"}`,
			wantErr: "cannot unmarshal string",
		},
		{
			name:    "invalid json",
			body:    `not json`,
			wantErr: "invalid character",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := decodeSingleQuestionAnswering([]byte(tc.body))
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)

				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestFeatureExtractionTokenCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := HuggingFaceFeatureExtractionMixin{}
	got, err := codec.FeatureExtractionTokenCodec().Decode(DecodeParams{
		Body:    []byte(`[[0.1,0.2],[0.3,0.4]]`),
		Headers: jsonHeaders(mimeTypeJSON),
	})
	require.NoError(t, err)
	require.Equal(t, hftypes.FeatureExtractionTokenEmbeddings{{0.1, 0.2}, {0.3, 0.4}}, got)
}

func TestFeatureExtractionTokenBatchCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := HuggingFaceFeatureExtractionMixin{}
	got, err := codec.FeatureExtractionTokenBatchCodec().Decode(DecodeParams{
		Body:    []byte(`[[[0.1],[0.2]],[[0.3]]]`),
		Headers: jsonHeaders(mimeTypeJSON),
	})
	require.NoError(t, err)
	require.Equal(
		t,
		[]hftypes.FeatureExtractionTokenEmbeddings{{{0.1}, {0.2}}, {{0.3}}},
		got,
	)
}

func TestQuestionAnsweringCodec_EncodePassesHeaders(t *testing.T) {
	t.Parallel()

	body, headers, err := HuggingFaceQuestionAnsweringMixin{}.QuestionAnsweringCodec().Encode(
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

func TestTextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	body, headers, err := HuggingFaceTextToImageMixin{}.TextToImageCodec().Encode(
		EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input: "a serene mountain landscape at sunset",
				Parameters: &hftypes.TextToImageParameters{
					GuidanceScale: new(7.5),
					Width:         new(512),
				},
			},
			Model: "test-model",
		},
	)
	require.NoError(t, err)
	require.JSONEq(
		t,
		`{"inputs":"a serene mountain landscape at sunset","parameters":{"guidance_scale":7.5,"width":512}}`,
		string(body),
	)
	require.Equal(t, ctJSON, headers.Get("Content-Type")) //nolint:testifylint // media type
	require.Equal(t, "*/*", headers.Get("Accept"))
}

func TestTextToImageCodec_Decode(t *testing.T) {
	t.Parallel()

	c := HuggingFaceTextToImageMixin{}.TextToImageCodec()
	imageBytes := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}

	cases := []struct {
		name        string
		contentType string
		wantMedia   string
		wantErr     bool
	}{
		{
			name:        "png",
			contentType: "image/png",
			wantMedia:   "image/png",
		},
		{
			name:        "jpeg",
			contentType: "image/jpeg",
			wantMedia:   "image/jpeg",
		},
		{
			name:        "webp",
			contentType: "image/webp",
			wantMedia:   "image/webp",
		},
		{
			name:        "image with charset parameter",
			contentType: "image/png; charset=utf-8",
			wantMedia:   "image/png",
		},
		{
			name:        "uppercase media type normalized",
			contentType: "IMAGE/PNG",
			wantMedia:   "image/png",
		},
		{
			name:        "json rejected",
			contentType: "application/json",
			wantErr:     true,
		},
		{
			name:        "text rejected",
			contentType: "text/plain",
			wantErr:     true,
		},
		{
			name:        "missing content type rejected",
			contentType: "",
			wantErr:     true,
		},
		{
			name:        "malformed content type rejected",
			contentType: "not a media type",
			wantErr:     true,
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := c.Decode(DecodeParams{
				Body:    imageBytes,
				Headers: jsonHeaders(tc.contentType),
			})

			if tc.wantErr {
				require.Error(t, err)
				assertSerializationError(t, err)
				require.Equal(t, hftypes.TextToImageResponse{}, got)

				return
			}

			require.NoError(t, err)
			require.Equal(t, imageBytes, got.Image)
			require.Equal(t, tc.wantMedia, got.MediaType)
		})
	}
}

func TestParseMediaType(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		contentType string
		wantMedia   string
		wantOK      bool
	}{
		{
			name:        "plain",
			contentType: "application/json",
			wantMedia:   "application/json",
			wantOK:      true,
		},
		{
			name:        "with charset parameter",
			contentType: "text/plain; charset=utf-8",
			wantMedia:   "text/plain",
			wantOK:      true,
		},
		{
			name:        "uppercase normalized",
			contentType: "IMAGE/PNG",
			wantMedia:   "image/png",
			wantOK:      true,
		},
		{
			name:        "empty",
			contentType: "",
			wantOK:      false,
		},
		{
			name:        "malformed",
			contentType: "not a media type",
			wantOK:      false,
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			media, ok := parseMediaType(tc.contentType)

			require.Equal(t, tc.wantOK, ok)
			require.Equal(t, tc.wantMedia, media)
		})
	}
}

// failingParameters fails to marshal, exercising the flatten step's marshal
// error branch.
type failingParameters struct{}

// MarshalJSON implements [json.Marshaler].
func (failingParameters) MarshalJSON() ([]byte, error) {
	return nil, errors.New("marshal failure")
}

// arrayParameters marshals to a JSON array, exercising the flatten step's
// unmarshal error branch: the payload cannot decode into a JSON object.
type arrayParameters struct{}

// MarshalJSON implements [json.Marshaler].
func (arrayParameters) MarshalJSON() ([]byte, error) {
	return []byte("[1,2,3]"), nil
}

// TestFlattenParameters exercises the shared parameter-flattening helper:
// nil parameters flatten to an empty map, marshal and flatten failures
// surface as serialization SDK errors, and valid parameters flatten to their
// JSON object form with exact numeric literals.
func TestFlattenParameters(t *testing.T) {
	t.Parallel()

	t.Run("nil parameters flatten to an empty map", func(t *testing.T) {
		t.Parallel()

		flat, err := flattenParameters(nil)
		require.NoError(t, err)
		require.Empty(t, flat)
	})

	t.Run("marshal failure is a serialization error", func(t *testing.T) {
		t.Parallel()

		_, err := flattenParameters(failingParameters{})
		assertSerializationError(t, err)
	})

	t.Run("non-object payload is a serialization error", func(t *testing.T) {
		t.Parallel()

		_, err := flattenParameters(arrayParameters{})
		assertSerializationError(t, err)
	})

	t.Run("valid parameters flatten", func(t *testing.T) {
		t.Parallel()

		flat, err := flattenParameters(hftypes.TextToVideoParameters{
			Seed: new(int64(42)),
		})
		require.NoError(t, err)
		require.Equal(t, map[string]any{"seed": json.Number("42")}, flat)
	})

	t.Run("large seed preserves exact numeric literal", func(t *testing.T) {
		t.Parallel()

		// A seed beyond float64's 2^53 integer precision must arrive intact
		// as a json.Number instead of being silently rounded.
		flat, err := flattenParameters(hftypes.TextToVideoParameters{
			Seed: new(int64(9223372036854775807)),
		})
		require.NoError(t, err)
		require.Equal(t, map[string]any{"seed": json.Number("9223372036854775807")}, flat)
	})
}

// TestMergeInferenceParameters exercises the merge helper: nil parameters
// leave the payload untouched, valid parameters merge in, and flatten failures
// surface as errors.
func TestMergeInferenceParameters(t *testing.T) {
	t.Parallel()

	t.Run("nil parameters leave payload untouched", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"existing": "value"}
		err := mergeInferenceParameters(payload, nil)
		require.NoError(t, err)
		require.Equal(t, map[string]any{"existing": "value"}, payload)
	})

	t.Run("parameters merge into payload", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"existing": "value"}
		err := mergeInferenceParameters(payload, hftypes.TextToVideoParameters{
			Seed: new(int64(42)),
		})
		require.NoError(t, err)
		require.Equal(t, map[string]any{
			"existing": "value",
			"seed":     json.Number("42"),
		}, payload)
	})

	t.Run("flatten failure surfaces", func(t *testing.T) {
		t.Parallel()

		err := mergeInferenceParameters(map[string]any{}, failingParameters{})
		assertSerializationError(t, err)
	})
}

// TestMergeRequestParameters exercises the request-parameter merge: typed
// parameters merge first, provider parameters override them, and a flatten
// failure from the typed parameters surfaces before provider parameters apply.
func TestMergeRequestParameters(t *testing.T) {
	t.Parallel()

	t.Run("provider parameters override typed parameters", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{}
		err := mergeRequestParameters(payload, hftypes.TextToVideoParameters{
			Seed: new(int64(42)),
		}, map[string]any{"seed": 7, "extra": "value"})
		require.NoError(t, err)
		require.Equal(t, map[string]any{"seed": 7, "extra": "value"}, payload)
	})

	t.Run("nil typed parameters merge provider parameters only", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"existing": "value"}
		err := mergeRequestParameters(payload, nil, map[string]any{"extra": "value"})
		require.NoError(t, err)
		require.Equal(t, map[string]any{"existing": "value", "extra": "value"}, payload)
	})

	t.Run("typed parameter flatten failure surfaces", func(t *testing.T) {
		t.Parallel()

		err := mergeRequestParameters(
			map[string]any{},
			failingParameters{},
			map[string]any{"extra": "value"},
		)
		assertSerializationError(t, err)
	})
}

func TestTextClassificationCodec_Decode(t *testing.T) {
	t.Parallel()

	c := HuggingFaceTextClassificationMixin{}.TextClassificationCodec()
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
