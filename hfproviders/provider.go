// Package hfproviders defines the Provider interface and built-in provider implementations.
package hfproviders

import (
	"context"
	"fmt"
	"reflect"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// AsProvider casts the given [Provider] implementation into the desired
// [Provider] type, e.g. [SummarizationProvider], [ChatProvider], etc.
// If provider is nil (including a typed-nil pointer wrapped in the interface)
// or does not implement the desired provider type, an error is returned
// stating so.
func AsProvider[T Provider](provider Provider) (T, error) {
	if utils.IsNil(provider) {
		var zero T

		return zero, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "provider must not be nil",
			Err:     nil,
		}
	}

	tProvider, ok := provider.(T)
	if !ok {
		return tProvider, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf(
				"%s does not implement %s",
				provider.Name(),
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
	// ProviderSuffix returns the routing token this provider wants appended to
	// model IDs for wire formats that carry the model in the request body. It
	// is a general model-routing capability rather than a chat-specific one, so
	// any task that adopts a model-in-body format can reuse it. An empty string
	// means no suffix is appended (e.g. for the default HuggingFace provider,
	// where the router selects the provider server-side).
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

// Endpoint describes a resolved task endpoint: the HTTP method to use and the
// path to send it to, relative to Options.BaseURL. An empty Method is
// defaulted to POST by the typed dispatch layer.
type Endpoint struct {
	// Method is the HTTP request method, e.g. [net/http.MethodPost].
	// Providers should use the http.Method* constants. An empty Method is
	// treated as POST.
	Method string

	// Path is the request path relative to Options.BaseURL. Absolute paths
	// are rejected by the transport so a provider can never redirect a
	// request away from the caller's configured host.
	Path string
}

// ChatProvider is a Provider that supplies an endpoint and Codec for the Chat task.
type ChatProvider interface {
	Provider
	ChatEndpoint(params EndpointParams) (Endpoint, error)
	ChatCodec() ChatCodec
}

// ChatStreamProvider is a Provider that supplies an endpoint and Codec for the ChatStream task.
type ChatStreamProvider interface {
	Provider
	ChatEndpoint(params EndpointParams) (Endpoint, error)
	ChatStreamCodec() ChatStreamCodec
}

// FeatureExtractionBatchProvider is a Provider that supplies an endpoint and Codec for the
// FeatureExtractionBatch task.
type FeatureExtractionBatchProvider interface {
	Provider
	FeatureExtractionBatchEndpoint(params EndpointParams) (Endpoint, error)
	FeatureExtractionBatchCodec() FeatureExtractionBatchCodec
}

// FeatureExtractionProvider is a Provider that supplies an endpoint and Codec for the
// FeatureExtraction task.
type FeatureExtractionProvider interface {
	Provider
	FeatureExtractionEndpoint(params EndpointParams) (Endpoint, error)
	FeatureExtractionCodec() FeatureExtractionCodec
}

// FillMaskProvider is a Provider that supplies an endpoint and Codec for the FillMask task.
type FillMaskProvider interface {
	Provider
	FillMaskEndpoint(params EndpointParams) (Endpoint, error)
	FillMaskCodec() FillMaskCodec
}

// ImageClassificationProvider is a Provider that supplies an endpoint and Codec
// for the ImageClassification task.
type ImageClassificationProvider interface {
	Provider
	ImageClassificationEndpoint(params EndpointParams) (Endpoint, error)
	ImageClassificationCodec() ImageClassificationCodec
}

// ObjectDetectionProvider is a Provider that supplies an endpoint and Codec for
// the ObjectDetection task.
type ObjectDetectionProvider interface {
	Provider
	ObjectDetectionEndpoint(params EndpointParams) (Endpoint, error)
	ObjectDetectionCodec() ObjectDetectionCodec
}

// ImageSegmentationProvider is a Provider that supplies an endpoint and Codec
// for the ImageSegmentation task.
type ImageSegmentationProvider interface {
	Provider
	ImageSegmentationEndpoint(params EndpointParams) (Endpoint, error)
	ImageSegmentationCodec() ImageSegmentationCodec
}

// QuestionAnsweringProvider is a Provider that supplies an endpoint and Codec
// for the QuestionAnswering task.
type QuestionAnsweringProvider interface {
	Provider
	QuestionAnsweringEndpoint(params EndpointParams) (Endpoint, error)
	QuestionAnsweringCodec() QuestionAnsweringCodec
}

// SummarizationProvider is a Provider that supplies an endpoint and Codec for
// the Summarization task.
type SummarizationProvider interface {
	Provider
	SummarizationEndpoint(params EndpointParams) (Endpoint, error)
	SummarizationCodec() SummarizationCodec
}

// TableQuestionAnsweringProvider is a Provider that supplies an endpoint and
// Codec for the TableQuestionAnswering task.
type TableQuestionAnsweringProvider interface {
	Provider
	TableQuestionAnsweringEndpoint(params EndpointParams) (Endpoint, error)
	TableQuestionAnsweringCodec() TableQuestionAnsweringCodec
}

// TextClassificationProvider is a Provider that supplies an endpoint and Codec
// for the TextClassification task.
type TextClassificationProvider interface {
	Provider
	TextClassificationEndpoint(params EndpointParams) (Endpoint, error)
	TextClassificationCodec() TextClassificationCodec
}

// TextToImageProvider is a Provider that supplies an endpoint and Codec for the
// TextToImage task.
type TextToImageProvider interface {
	Provider
	TextToImageEndpoint(params EndpointParams) (Endpoint, error)
	TextToImageCodec() TextToImageCodec
}

// TokenClassificationProvider is a Provider that supplies an endpoint and Codec
// for the TokenClassification task.
type TokenClassificationProvider interface {
	Provider
	TokenClassificationEndpoint(params EndpointParams) (Endpoint, error)
	TokenClassificationCodec() TokenClassificationCodec
}

// TranslationProvider is a Provider that supplies an endpoint and Codec for the
// Translation task.
type TranslationProvider interface {
	Provider
	TranslationEndpoint(params EndpointParams) (Endpoint, error)
	TranslationCodec() TranslationCodec
}

// ZeroShotTextClassificationProvider is a Provider that supplies an endpoint
// and Codec for the ZeroShotTextClassification task.
type ZeroShotTextClassificationProvider interface {
	Provider
	ZeroShotTextClassificationEndpoint(params EndpointParams) (Endpoint, error)
	ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec
}
