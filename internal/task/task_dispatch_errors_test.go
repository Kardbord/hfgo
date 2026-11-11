//go:build !integration

package task_test

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testproviders"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// noTaskProvider implements only the base hfproviders.Provider interface.
// Passing it to any typed task dispatch must fail the AsProvider cast with a
// configuration error, before a request is made.
type noTaskProvider struct{}

func (noTaskProvider) Name() string { return "no-task" }

func (noTaskProvider) ProviderSuffix() string { return "" }

// errEndpointUnavailable is the sentinel returned by every endpoint method of
// endpointFailProvider.
var errEndpointUnavailable = errors.New("endpoint unavailable")

// endpointFailProvider embeds the shared MockProvider (Hugging Face codec and
// endpoint defaults) and shadows every endpoint method to fail, so each typed
// dispatch can be exercised through its endpoint-error wrapping path.
type endpointFailProvider struct {
	testproviders.MockProvider
}

func (p endpointFailProvider) ChatEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ObjectDetectionEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ImageClassificationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ImageSegmentationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TextToImageEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ImageToImageEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

// ImageToImageCodec satisfies [hfproviders.ImageToImageProvider]. The
// endpoint-error sweep short-circuits before Encode, so the codec is never
// invoked.
func (p endpointFailProvider) ImageToImageCodec() hfproviders.ImageToImageCodec {
	return hfproviders.JSONCodec[hftypes.ImageToImageRequest, hftypes.ImageToImageResponse]{}
}

func (p endpointFailProvider) TextToVideoEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

// TextToVideoCodec satisfies [hfproviders.TextToVideoProvider]. The
// endpoint-error sweep short-circuits before Encode, so the codec is never
// invoked.
func (p endpointFailProvider) TextToVideoCodec() hfproviders.TextToVideoCodec {
	return hfproviders.JSONCodec[hftypes.TextToVideoRequest, hftypes.TextToVideoResponse]{}
}

func (p endpointFailProvider) FeatureExtractionEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FeatureExtractionBatchEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TextClassificationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ZeroShotTextClassificationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TokenClassificationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) QuestionAnsweringEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TableQuestionAnsweringEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FillMaskEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) SummarizationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TranslationEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) SpeechRecognitionEndpoint(
	_ context.Context,
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func chatDispatchRequest() hftypes.ChatRequest {
	return hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hi")}},
		},
	}
}

func tableQuestionAnsweringDispatchRequest() hftypes.TableQuestionAnsweringRequest {
	return hftypes.TableQuestionAnsweringRequest{
		Input: hftypes.TableQuestionAnsweringInput{
			Question: "How old is Bob?",
			Table: map[string][]string{
				"Name": {"Alice", "Bob", "Carol"},
				"Age":  {"25", "30", "35"},
			},
		},
	}
}

// dispatchErrorCase describes one typed task dispatch and the prefix its
// endpoint-error wrapping uses. The same table drives both the AsProvider and
// endpoint-failure sweeps so every task keeps both paths covered in one
// place.
type dispatchErrorCase struct {
	// name is the subtest name.
	name string
	// wantPrefix is the task-specific endpoint-error wrapping prefix.
	wantPrefix string
	// call issues a single dispatch through the public client.
	call func(opts ...hfopts.Option) (any, error)
}

func dispatchErrorCases() []dispatchErrorCase {
	return []dispatchErrorCase{
		{
			"Chat",
			"error computing chat endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).Chat(context.Background(), chatDispatchRequest())
			},
		},
		{
			"ChatStream",
			"error computing chat streaming endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ChatStream(context.Background(), chatDispatchRequest())
			},
		},
		{
			"ExtractPooledFeatures",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ExtractPooledFeatures(context.Background(), hftypes.FeatureExtractionRequest{Input: "test input"})
			},
		},
		{
			"ExtractPooledFeaturesBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ExtractPooledFeaturesBatch(context.Background(), hftypes.FeatureExtractionBatchRequest{Inputs: []string{"hi"}})
			},
		},
		{
			"ExtractTokenFeatures",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ExtractTokenFeatures(context.Background(), hftypes.FeatureExtractionRequest{Input: "test input"})
			},
		},
		{
			"ExtractTokenFeaturesBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ExtractTokenFeaturesBatch(
						context.Background(),
						hftypes.FeatureExtractionBatchRequest{Inputs: []string{"hi"}},
					)
			},
		},
		{
			"FillMask",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					FillMask(context.Background(), hftypes.FillMaskRequest{Input: "The cat is [MASK]."})
			},
		},
		{
			"Summarize",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					Summarize(context.Background(), hftypes.SummarizationRequest{Input: "Some long text."})
			},
		},
		{
			"ClassifyText",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ClassifyText(context.Background(), hftypes.TextClassificationRequest{Input: "test text"})
			},
		},
		{
			"ClassifyTokens",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ClassifyTokens(context.Background(), hftypes.TokenClassificationRequest{Input: "My name is Sarah."})
			},
		},
		{
			"Translate",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					Translate(context.Background(), hftypes.TranslationRequest{Input: "Hello."})
			},
		},
		{
			"AnswerQuestion",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).AnswerQuestion(
					context.Background(),
					hftypes.QuestionAnsweringRequest{
						Input: hftypes.QuestionAnsweringInput{
							Question: "What is the capital of France?",
							Context:  "France is a country in Europe.",
						},
					},
				)
			},
		},
		{
			"AnswerTableQuestion",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).AnswerTableQuestion(
					context.Background(),
					tableQuestionAnsweringDispatchRequest(),
				)
			},
		},
		{
			"ZeroShotClassifyText",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				req := hftypes.ZeroShotTextClassificationRequest{
					Input: "test text",
					Parameters: &hftypes.ZeroShotTextClassificationParameters{
						CandidateLabels: []string{"positive", "negative"},
					},
				}

				return hfgo.NewClient(opts...).ZeroShotClassifyText(context.Background(), req)
			},
		},
		{
			"DetectObjects",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).DetectObjects(
					context.Background(),
					hftypes.ObjectDetectionRequest{Input: testutils.TinyPNGBase64},
				)
			},
		},
		{
			"ClassifyImage",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).ClassifyImage(
					context.Background(),
					hftypes.ImageClassificationRequest{Input: testutils.TinyPNGBase64},
				)
			},
		},
		{
			"SegmentImage",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).SegmentImage(
					context.Background(),
					hftypes.ImageSegmentationRequest{Input: testutils.TinyPNGBase64},
				)
			},
		},
		{
			"GenerateImage",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).GenerateImage(
					context.Background(),
					hftypes.TextToImageRequest{Input: "a cat"},
				)
			},
		},
		{
			"GenerateImageToImage",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).GenerateImageToImage(
					context.Background(),
					hftypes.ImageToImageRequest{Input: testutils.TinyPNGBase64},
				)
			},
		},
		{
			"GenerateVideo",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).GenerateVideo(
					context.Background(),
					hftypes.TextToVideoRequest{Input: "a cat"},
				)
			},
		},
		{
			"RecognizeSpeech",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).RecognizeSpeech(
					context.Background(),
					hftypes.SpeechRecognitionRequest{Input: "AAAA"},
				)
			},
		},
	}
}

// TestDispatch_AsProviderErrors sweeps every typed task dispatch with a
// provider that satisfies only the base Provider interface, asserting each
// fails the AsProvider cast with a configuration error before any request.
func TestDispatch_AsProviderErrors(t *testing.T) {
	t.Parallel()

	cases := dispatchErrorCases()
	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, `{}`, nil)
			result, err := tc.call(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
				hfopts.WithProvider(noTaskProvider{}),
			)
			require.Error(
				t,
				err,
				"AsProvider must reject a provider without the task interface",
			)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			require.True(
				t,
				reflect.ValueOf(result).IsZero(),
				"expected zero result, got %v",
				result,
			)
			require.Nil(t, mt.LastRequest, "AsProvider failure short-circuits before any request")
		})
	}
}

// TestDispatch_EndpointErrors sweeps every typed task dispatch with a
// provider whose endpoint computations all fail, asserting each wraps the
// failure into a configuration SDKError and dispatches no request.
func TestDispatch_EndpointErrors(t *testing.T) {
	t.Parallel()

	cases := dispatchErrorCases()
	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, `{}`, nil)
			result, err := tc.call(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
				hfopts.WithProvider(endpointFailProvider{}),
			)
			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			require.ErrorContains(t, err, tc.wantPrefix)
			require.ErrorContains(t, err, errEndpointUnavailable.Error())
			require.True(
				t,
				reflect.ValueOf(result).IsZero(),
				"expected zero result, got %v",
				result,
			)
			require.Nil(t, mt.LastRequest, "endpoint failure short-circuits before any request")
		})
	}
}
