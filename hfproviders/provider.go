// Package hfproviders defines the Provider interface and built-in provider implementations.
package hfproviders

import (
	"context"
	"fmt"
	"reflect"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// AsProvider casts the given [Provider] implementation into the desired
// [Provider] type, e.g. [SummarizationProvider], [ChatProvider], etc.
// If the cast fails, an error is returned stating that the given provider
// does not implement the desired provider type.
func AsProvider[T Provider](p Provider) (T, error) {
	tProvider, ok := p.(T)
	if !ok {
		name := "<nil>"
		if p != nil {
			name = p.Name()
		}

		return tProvider, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf(
				"%s does not implement %s",
				name,
				reflect.TypeFor[T](),
			),
			Err: nil,
		}
	}

	return tProvider, nil
}

// Provider knows how to construct API endpoints and transform between HF-spec
// and provider-spec wire formats for a given task and model. Implementations
// must be safe for concurrent use.
//
// Endpoint methods return paths relative to the caller-configured
// Options.BaseURL. The transport rejects absolute paths, so a provider can
// never redirect a request (and its bearer token) to a host the caller did
// not choose. Authentication is owned by the caller via Options.Token and
// Options.Headers; provider and codec implementations must not set an
// Authorization header.
type Provider interface {
	// ProviderSuffix returns the provider suffix appended to model IDs
	// for routing in OpenAI-compatible endpoints. An empty string means
	// no suffix is appended (e.g. for the default HuggingFace provider).
	ProviderSuffix() string

	// Name returns the name of the given provider for debug purposes.
	Name() string
}

// EndpointParams carries the inputs to a per-task endpoint resolution method,
// e.g. [ChatProvider.ChatEndpoint]. The typed dispatch layer guarantees a
// non-empty Model before any endpoint method is called.
type EndpointParams struct {
	// Context carries the caller's request context, including its deadline
	// and cancellation signals. It is never nil (it falls back to
	// context.Background()). Implementations must not retain it beyond the
	// duration of the call.
	//
	//nolint:containedctx // EndpointParams is a per-call carrier, not stored state.
	Context context.Context

	// Model identifies the model the request is routed to. For chat tasks
	// it is the resolved routing identifier including any provider suffix
	// (e.g. "mistral-7b:sambanova"), because chat payloads carry the model
	// themselves. For pipeline tasks it is the configured model ID; routing
	// there is expressed in the endpoint path. Model-dependent endpoints
	// may validate it defensively; model-independent endpoints (e.g.
	// ChatEndpoint) must not reject an empty Model.
	Model string
}

// ChatProvider is a Provider that supplies an endpoint and Codec for the Chat task.
type ChatProvider interface {
	Provider
	ChatEndpoint(params EndpointParams) (string, error)
	ChatCodec() ChatCodec
}

// ChatStreamProvider is a Provider that supplies an endpoint and Codec for the ChatStream task.
type ChatStreamProvider interface {
	Provider
	ChatEndpoint(params EndpointParams) (string, error)
	ChatStreamCodec() ChatStreamCodec
}

// FeatureExtractionBatchProvider is a Provider that supplies an endpoint and Codec for the
// FeatureExtractBatch task.
type FeatureExtractionBatchProvider interface {
	Provider
	FeatureExtractionBatchEndpoint(params EndpointParams) (string, error)
	FeatureExtractionBatchCodec() FeatureExtractionBatchCodec
}

// FeatureExtractionProvider is a Provider that supplies an endpoint and Codec for the
// FeatureExtract task.
type FeatureExtractionProvider interface {
	Provider
	FeatureExtractionEndpoint(params EndpointParams) (string, error)
	FeatureExtractionCodec() FeatureExtractionCodec
}

// FillMaskBatchProvider is a Provider that supplies an endpoint and Codec for the FillMaskBatch task.
type FillMaskBatchProvider interface {
	Provider
	FillMaskBatchEndpoint(params EndpointParams) (string, error)
	FillMaskBatchCodec() FillMaskBatchCodec
}

// FillMaskProvider is a Provider that supplies an endpoint and Codec for the FillMask task.
type FillMaskProvider interface {
	Provider
	FillMaskEndpoint(params EndpointParams) (string, error)
	FillMaskCodec() FillMaskCodec
}

// QuestionAnsweringProvider is a Provider that supplies an endpoint and Codec for the
// AnswerQuestion task.
type QuestionAnsweringProvider interface {
	Provider
	QuestionAnsweringEndpoint(params EndpointParams) (string, error)
	QuestionAnsweringCodec() QuestionAnsweringCodec
}

// SummarizationBatchProvider is a Provider that supplies an endpoint and Codec for the
// SummarizeBatch task.
type SummarizationBatchProvider interface {
	Provider
	SummarizationBatchEndpoint(params EndpointParams) (string, error)
	SummarizationBatchCodec() SummarizationBatchCodec
}

// SummarizationProvider is a Provider that supplies an endpoint and Codec for the Summarize task.
type SummarizationProvider interface {
	Provider
	SummarizationEndpoint(params EndpointParams) (string, error)
	SummarizationCodec() SummarizationCodec
}

// TableQuestionAnsweringProvider is a Provider that supplies an endpoint and Codec for the
// AnswerTableQuestion task.
type TableQuestionAnsweringProvider interface {
	Provider
	TableQuestionAnsweringEndpoint(params EndpointParams) (string, error)
	TableQuestionAnsweringCodec() TableQuestionAnsweringCodec
}

// TextClassificationBatchProvider is a Provider that supplies an endpoint and Codec for the
// ClassifyTextBatch task.
type TextClassificationBatchProvider interface {
	Provider
	TextClassificationBatchEndpoint(params EndpointParams) (string, error)
	TextClassificationBatchCodec() TextClassificationBatchCodec
}

// TextClassificationProvider is a Provider that supplies an endpoint and Codec for the
// ClassifyText task.
type TextClassificationProvider interface {
	Provider
	TextClassificationEndpoint(params EndpointParams) (string, error)
	TextClassificationCodec() TextClassificationCodec
}

// TokenClassificationBatchProvider is a Provider that supplies an endpoint and Codec for the
// ClassifyTokensBatch task.
type TokenClassificationBatchProvider interface {
	Provider
	TokenClassificationBatchEndpoint(params EndpointParams) (string, error)
	TokenClassificationBatchCodec() TokenClassificationBatchCodec
}

// TokenClassificationProvider is a Provider that supplies an endpoint and Codec for the
// ClassifyTokens task.
type TokenClassificationProvider interface {
	Provider
	TokenClassificationEndpoint(params EndpointParams) (string, error)
	TokenClassificationCodec() TokenClassificationCodec
}

// TranslationBatchProvider is a Provider that supplies an endpoint and Codec for the
// TranslateBatch task.
type TranslationBatchProvider interface {
	Provider
	TranslationBatchEndpoint(params EndpointParams) (string, error)
	TranslationBatchCodec() TranslationBatchCodec
}

// TranslationProvider is a Provider that supplies an endpoint and Codec for the Translate task.
type TranslationProvider interface {
	Provider
	TranslationEndpoint(params EndpointParams) (string, error)
	TranslationCodec() TranslationCodec
}

// ZeroShotTextClassificationBatchProvider is a Provider that supplies an endpoint and Codec
// for the ZeroShotClassifyTextBatch task.
type ZeroShotTextClassificationBatchProvider interface {
	Provider
	ZeroShotTextClassificationBatchEndpoint(params EndpointParams) (string, error)
	ZeroShotTextClassificationBatchCodec() ZeroShotTextClassificationBatchCodec
}

// ZeroShotTextClassificationProvider is a Provider that supplies an endpoint and Codec for
// the ZeroShotClassifyText task.
type ZeroShotTextClassificationProvider interface {
	Provider
	ZeroShotTextClassificationEndpoint(params EndpointParams) (string, error)
	ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec
}
