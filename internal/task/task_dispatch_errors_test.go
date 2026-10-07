//go:build !integration

package task_test

import (
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
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
	testutils.MockProvider
}

func (p endpointFailProvider) ChatEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FeatureExtractionEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FeatureExtractionBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TextClassificationEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TextClassificationBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ZeroShotTextClassificationEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) ZeroShotTextClassificationBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TokenClassificationEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TokenClassificationBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) QuestionAnsweringEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TableQuestionAnsweringEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FillMaskEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) FillMaskBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) SummarizationEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) SummarizationBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TranslationEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func (p endpointFailProvider) TranslationBatchEndpoint(
	_ hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{}, errEndpointUnavailable
}

func chatDispatchRequest() hftypes.ChatRequest {
	return hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: testutils.Ptr("hi")}},
		},
	}
}

func tableQADispatchRequest() hftypes.TableQuestionAnsweringRequest {
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
				return hfgo.NewClient(opts...).Chat(chatDispatchRequest())
			},
		},
		{
			"ChatStream",
			"error computing chat streaming endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).ChatStream(chatDispatchRequest())
			},
		},
		{
			"FeatureExtract",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					FeatureExtract(hftypes.FeatureExtractionRequest{Input: "test input"})
			},
		},
		{
			"FeatureExtractBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					FeatureExtractBatch(hftypes.FeatureExtractionBatchRequest{Inputs: []string{"hi"}})
			},
		},
		{
			"FillMask",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					FillMask(hftypes.FillMaskRequest{Input: "The cat is [MASK]."})
			},
		},
		{
			"FillMaskBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					FillMaskBatch(hftypes.FillMaskBatchRequest{Inputs: []string{"The cat is [MASK]."}})
			},
		},
		{
			"Summarize",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					Summarize(hftypes.SummarizationRequest{Input: "Some long text."})
			},
		},
		{
			"SummarizeBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					SummarizeBatch(hftypes.SummarizationBatchRequest{Inputs: []string{"Some long text."}})
			},
		},
		{
			"ClassifyText",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ClassifyText(hftypes.TextClassificationRequest{Input: "test text"})
			},
		},
		{
			"ClassifyTextBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ClassifyTextBatch(hftypes.TextClassificationBatchRequest{Inputs: []string{"test"}})
			},
		},
		{
			"ClassifyTokens",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					ClassifyTokens(hftypes.TokenClassificationRequest{Input: "My name is Sarah."})
			},
		},
		{
			"ClassifyTokensBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).ClassifyTokensBatch(
					hftypes.TokenClassificationBatchRequest{Inputs: []string{"My name is Sarah."}},
				)
			},
		},
		{
			"Translate",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					Translate(hftypes.TranslationRequest{Input: "Hello."})
			},
		},
		{
			"TranslateBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).
					TranslateBatch(hftypes.TranslationBatchRequest{Inputs: []string{"Hello."}})
			},
		},
		{
			"AnswerQuestion",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				return hfgo.NewClient(opts...).AnswerQuestion(
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
				return hfgo.NewClient(opts...).AnswerTableQuestion(tableQADispatchRequest())
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

				return hfgo.NewClient(opts...).ZeroShotClassifyText(req)
			},
		},
		{
			"ZeroShotClassifyTextBatch",
			"error computing endpoint:",
			func(opts ...hfopts.Option) (any, error) {
				req := hftypes.ZeroShotTextClassificationBatchRequest{
					Inputs: []string{"test text"},
					Parameters: &hftypes.ZeroShotTextClassificationParameters{
						CandidateLabels: []string{"positive", "negative"},
					},
				}

				return hfgo.NewClient(opts...).ZeroShotClassifyTextBatch(req)
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
