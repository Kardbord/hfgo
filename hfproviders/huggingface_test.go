//go:build !integration

package hfproviders

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestHuggingFaceProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	p := NewHuggingFaceProvider()
	require.Equal(t, "huggingface", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

// endpointCase pairs a display name with an endpoint method so every
// HuggingFaceProvider endpoint can be exercised through one table.
type endpointCase struct {
	name string
	call func(EndpointParams) (Endpoint, error)
	// wantPath computes the expected path from the model.
	wantPath func(model string) string
	// modelRequired reports whether the endpoint rejects an empty model
	// with a configuration error. Only the fixed chat path tolerates one.
	modelRequired bool
}

func endpointCases() []endpointCase {
	modelPath := func(m string) string { return hfModelPrefix + m }

	e := NewHuggingFaceProvider()
	pipeline := func(m string) string { return hfModelPrefix + m + "/pipeline/feature-extraction" }

	// ep builds a model-dependent case for the endpoints whose path is the
	// bare model prefix.
	ep := func(
		name string,
		call func(EndpointParams) (Endpoint, error),
	) endpointCase {
		return endpointCase{name: name, call: call, wantPath: modelPath, modelRequired: true}
	}

	return []endpointCase{
		{
			name:     "ChatEndpoint",
			call:     e.ChatEndpoint,
			wantPath: func(string) string { return "v1/chat/completions" },
		},
		{
			name:          "FeatureExtractionEndpoint",
			call:          e.FeatureExtractionEndpoint,
			wantPath:      pipeline,
			modelRequired: true,
		},
		{
			name:          "FeatureExtractionBatchEndpoint",
			call:          e.FeatureExtractionBatchEndpoint,
			wantPath:      pipeline,
			modelRequired: true,
		},
		ep("ObjectDetectionEndpoint", e.ObjectDetectionEndpoint),
		ep("ImageClassificationEndpoint", e.ImageClassificationEndpoint),
		ep("ImageSegmentationEndpoint", e.ImageSegmentationEndpoint),
		ep("TextClassificationEndpoint", e.TextClassificationEndpoint),
		ep("ZeroShotTextClassificationEndpoint", e.ZeroShotTextClassificationEndpoint),
		ep("TokenClassificationEndpoint", e.TokenClassificationEndpoint),
		ep("QuestionAnsweringEndpoint", e.QuestionAnsweringEndpoint),
		ep("TableQuestionAnsweringEndpoint", e.TableQuestionAnsweringEndpoint),
		ep("TextToImageEndpoint", e.TextToImageEndpoint),
		ep("FillMaskEndpoint", e.FillMaskEndpoint),
		ep("SummarizationEndpoint", e.SummarizationEndpoint),
		ep("TranslationEndpoint", e.TranslationEndpoint),
		ep("SpeechRecognitionEndpoint", e.SpeechRecognitionEndpoint),
	}
}

func TestHuggingFaceProvider_ValidModel(t *testing.T) {
	t.Parallel()

	for _, tc := range endpointCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ep, err := tc.call(EndpointParams{Model: "org/test-model"})
			require.NoError(t, err)
			require.Equal(t, http.MethodPost, ep.Method)
			require.Equal(t, tc.wantPath("org/test-model"), ep.Path)
		})
	}
}

func TestHuggingFaceProvider_EmptyModel(t *testing.T) {
	t.Parallel()

	for _, tc := range endpointCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ep, err := tc.call(EndpointParams{Model: ""})
			if !tc.modelRequired {
				require.NoError(t, err)
				require.Equal(t, tc.wantPath(""), ep.Path)

				return
			}

			require.Error(t, err)
			var sdkErr *hferrors.SDKError
			require.ErrorAs(t, err, &sdkErr)
			require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
			require.ErrorContains(t, err, errModelIsRequired)
		})
	}
}

// TestHuggingFaceProvider_CodecWiring asserts every codec accessor returns the
// expected built-in codec implementation, pinning the type wiring of the
// provider and its mixins.
func TestHuggingFaceProvider_CodecWiring(t *testing.T) {
	t.Parallel()

	c := NewHuggingFaceProvider()

	require.IsType(t, JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}, c.ChatCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		c.ChatStreamCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{},
		c.ObjectDetectionCodec(),
	)
	require.IsType(
		t,
		JSONCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionPooledEmbedding,
		]{},
		c.FeatureExtractionPooledBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionPooledEmbedding,
		]{},
		c.FeatureExtractionPooledCodec(),
	)
	require.IsType(
		t,
		JSONCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		c.FeatureExtractionTokenBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		c.FeatureExtractionTokenCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{},
		c.FillMaskCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ImageClassificationRequest, []hftypes.ImageClassification]{},
		c.ImageClassificationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]{},
		c.ImageSegmentationCodec(),
	)
	require.IsType(t, hfQuestionAnsweringCodec{}, c.QuestionAnsweringCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{},
		c.SummarizationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{},
		c.TableQuestionAnsweringCodec(),
	)
	require.IsType(t, hfTextClassificationCodec{}, c.TextClassificationCodec())
	require.IsType(t, hfTextToImageCodec{}, c.TextToImageCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{},
		c.TokenClassificationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{},
		c.TranslationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{},
		c.ZeroShotTextClassificationCodec(),
	)
	require.IsType(t, hfSpeechRecognitionCodec{}, c.SpeechRecognitionCodec())
}

func TestSpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	type encodeCase struct {
		name     string
		request  hftypes.SpeechRecognitionRequest
		validate func(*testing.T, []byte, http.Header, error)
	}

	cases := []encodeCase{
		{
			name:    "no params",
			request: hftypes.SpeechRecognitionRequest{Input: "AAAA"},
			validate: func(t *testing.T, body []byte, headers http.Header, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, body)
				require.Len(t, body, 3)
				require.Equal(t, byte(0), body[0])
				require.Equal(t, byte(0), body[1])
				require.Equal(t, byte(0), body[2])
				require.NotEmpty(t, headers.Get("Content-Type"))
			},
		},
		{
			name: "with params",
			request: hftypes.SpeechRecognitionRequest{
				Input: "AAAA",
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
			},
			validate: func(t *testing.T, body []byte, headers http.Header, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, body)
				require.Equal(t, "application/json", headers.Get("Content-Type"))

				var payload map[string]any
				err = json.Unmarshal(body, &payload)
				require.NoError(t, err)
				inputs, ok := payload["inputs"].(string)
				require.True(t, ok)
				require.Equal(t, "AAAA", inputs)
			},
		},
		{
			name:    "invalid base64",
			request: hftypes.SpeechRecognitionRequest{Input: "!!!invalid"},
			validate: func(t *testing.T, _ []byte, _ http.Header, err error) {
				t.Helper()
				require.Error(t, err)
				var sdkErr *hferrors.SDKError
				require.ErrorAs(t, err, &sdkErr)
				require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
			},
		},
	}

	codec := hfSpeechRecognitionCodec{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
				Request: tc.request,
			})
			tc.validate(t, body, headers, err)
		})
	}
}

func TestSpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := hfSpeechRecognitionCodec{}
	resp, err := codec.Decode(DecodeParams{
		Body:    []byte(`{"text":"hello world"}`),
		Headers: http.Header{"Content-Type": {"application/json"}},
	})
	require.NoError(t, err)
	require.Equal(t, "hello world", resp.Text)
	require.Nil(t, resp.Chunks)
}

// TestHuggingFaceProvider_ImplementsEveryTaskProvider pins the aggregate's
// method promotion. Because every per-task method name is unique to its mixin,
// these assertions also confirm each mixin carries the methods its interface
// requires.
func TestHuggingFaceProvider_ImplementsEveryTaskProvider(t *testing.T) {
	t.Parallel()

	var (
		_ ChatProvider                         = HuggingFaceProvider{}
		_ ChatStreamProvider                   = HuggingFaceProvider{}
		_ FeatureExtractionPooledBatchProvider = HuggingFaceProvider{}
		_ FeatureExtractionPooledProvider      = HuggingFaceProvider{}
		_ FeatureExtractionTokenProvider       = HuggingFaceProvider{}
		_ FeatureExtractionTokenBatchProvider  = HuggingFaceProvider{}
		_ FillMaskProvider                     = HuggingFaceProvider{}
		_ ImageClassificationProvider          = HuggingFaceProvider{}
		_ ObjectDetectionProvider              = HuggingFaceProvider{}
		_ ImageSegmentationProvider            = HuggingFaceProvider{}
		_ QuestionAnsweringProvider            = HuggingFaceProvider{}
		_ SummarizationProvider                = HuggingFaceProvider{}
		_ TableQuestionAnsweringProvider       = HuggingFaceProvider{}
		_ TextClassificationProvider           = HuggingFaceProvider{}
		_ TextToImageProvider                  = HuggingFaceProvider{}
		_ TokenClassificationProvider          = HuggingFaceProvider{}
		_ TranslationProvider                  = HuggingFaceProvider{}
		_ ZeroShotTextClassificationProvider   = HuggingFaceProvider{}
		_ SpeechRecognitionProvider            = HuggingFaceProvider{}
	)
}
