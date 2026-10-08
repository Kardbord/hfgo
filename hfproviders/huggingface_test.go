//go:build !integration

package hfproviders

import (
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
// HuggingFaceEndpoints method can be exercised through one table.
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

	e := HuggingFaceEndpoints{}
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
		ep("DetectObjectsEndpoint", e.DetectObjectsEndpoint),
		ep("TextClassificationEndpoint", e.TextClassificationEndpoint),
		ep("TextClassificationBatchEndpoint", e.TextClassificationBatchEndpoint),
		ep("ZeroShotTextClassificationEndpoint", e.ZeroShotTextClassificationEndpoint),
		ep(
			"ZeroShotTextClassificationBatchEndpoint",
			e.ZeroShotTextClassificationBatchEndpoint,
		),
		ep("TokenClassificationEndpoint", e.TokenClassificationEndpoint),
		ep("TokenClassificationBatchEndpoint", e.TokenClassificationBatchEndpoint),
		ep("QuestionAnsweringEndpoint", e.QuestionAnsweringEndpoint),
		ep("TableQuestionAnsweringEndpoint", e.TableQuestionAnsweringEndpoint),
		ep("FillMaskEndpoint", e.FillMaskEndpoint),
		ep("FillMaskBatchEndpoint", e.FillMaskBatchEndpoint),
		ep("SummarizationEndpoint", e.SummarizationEndpoint),
		ep("SummarizationBatchEndpoint", e.SummarizationBatchEndpoint),
		ep("TranslationEndpoint", e.TranslationEndpoint),
		ep("TranslationBatchEndpoint", e.TranslationBatchEndpoint),
	}
}

func TestHuggingFaceEndpoints_ValidModel(t *testing.T) {
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

func TestHuggingFaceEndpoints_EmptyModel(t *testing.T) {
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

// TestHuggingFaceCodecs_Wiring asserts every codec accessor returns the
// expected built-in codec implementation, pinning the type wiring of the
// embeddable defaults.
func TestHuggingFaceCodecs_Wiring(t *testing.T) {
	t.Parallel()

	c := HuggingFaceCodecs{}

	require.IsType(t, JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}, c.ChatCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		c.ChatStreamCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{},
		c.DetectObjectsCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]{},
		c.FeatureExtractionBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]{},
		c.FeatureExtractionCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.FillMaskBatchRequest, [][]hftypes.FillMaskPrediction]{},
		c.FillMaskBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{},
		c.FillMaskCodec(),
	)
	require.IsType(t, hfQuestionAnsweringCodec{}, c.QuestionAnsweringCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.SummarizationBatchRequest, []hftypes.Summarization]{},
		c.SummarizationBatchCodec(),
	)
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
	require.IsType(
		t,
		JSONCodec[hftypes.TextClassificationBatchRequest, [][]hftypes.TextClassification]{},
		c.TextClassificationBatchCodec(),
	)
	require.IsType(t, hfTextClassificationCodec{}, c.TextClassificationCodec())
	require.IsType(
		t,
		JSONCodec[hftypes.TokenClassificationBatchRequest, [][]hftypes.TokenClassification]{},
		c.TokenClassificationBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{},
		c.TokenClassificationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.TranslationBatchRequest, []hftypes.Translation]{},
		c.TranslationBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{},
		c.TranslationCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ZeroShotTextClassificationBatchRequest, []hftypes.ZeroShotTextClassificationBatched]{},
		c.ZeroShotTextClassificationBatchCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{},
		c.ZeroShotTextClassificationCodec(),
	)
}
