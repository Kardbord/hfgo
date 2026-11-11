package hfproviders

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const (
	errModelIsRequired = "model is required"
	hfModelPrefix      = "hf-inference/models/"
)

// HuggingFaceProvider implements Provider for the Hugging Face inference API.
// It composes every built-in per-task mixin, so it serves all tasks with the
// Hugging Face endpoints and wire formats.
type HuggingFaceProvider struct {
	HuggingFaceChatMixin
	HuggingFaceFeatureExtractionMixin
	HuggingFaceFillMaskMixin
	HuggingFaceImageClassificationMixin
	HuggingFaceImageSegmentationMixin
	HuggingFaceObjectDetectionMixin
	HuggingFaceQuestionAnsweringMixin
	HuggingFaceSpeechRecognitionMixin
	HuggingFaceSummarizationMixin
	HuggingFaceTableQuestionAnsweringMixin
	HuggingFaceTextClassificationMixin
	HuggingFaceTextToImageMixin
	HuggingFaceTokenClassificationMixin
	HuggingFaceTranslationMixin
	HuggingFaceZeroShotTextClassificationMixin
}

// Compile-time assertions that HuggingFaceProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider                         = (*HuggingFaceProvider)(nil)
	_ ChatStreamProvider                   = (*HuggingFaceProvider)(nil)
	_ FeatureExtractionPooledBatchProvider = (*HuggingFaceProvider)(nil)
	_ FeatureExtractionPooledProvider      = (*HuggingFaceProvider)(nil)
	_ FeatureExtractionTokenProvider       = (*HuggingFaceProvider)(nil)
	_ FeatureExtractionTokenBatchProvider  = (*HuggingFaceProvider)(nil)
	_ FillMaskProvider                     = (*HuggingFaceProvider)(nil)
	_ ImageClassificationProvider          = (*HuggingFaceProvider)(nil)
	_ ImageSegmentationProvider            = (*HuggingFaceProvider)(nil)
	_ ObjectDetectionProvider              = (*HuggingFaceProvider)(nil)
	_ QuestionAnsweringProvider            = (*HuggingFaceProvider)(nil)
	_ SpeechRecognitionProvider            = (*HuggingFaceProvider)(nil)
	_ SummarizationProvider                = (*HuggingFaceProvider)(nil)
	_ TableQuestionAnsweringProvider       = (*HuggingFaceProvider)(nil)
	_ TextClassificationProvider           = (*HuggingFaceProvider)(nil)
	_ TextToImageProvider                  = (*HuggingFaceProvider)(nil)
	_ TokenClassificationProvider          = (*HuggingFaceProvider)(nil)
	_ TranslationProvider                  = (*HuggingFaceProvider)(nil)
	_ ZeroShotTextClassificationProvider   = (*HuggingFaceProvider)(nil)
)

// Compile-time assertions that the Hugging Face provider's custom codecs
// participate in their task pipelines. The rest reuse [JSONCodec] through
// their mixins and need no per-codec assertion.
var (
	_ QuestionAnsweringCodec  = (*hfQuestionAnsweringCodec)(nil)
	_ TextClassificationCodec = (*hfTextClassificationCodec)(nil)
	_ TextToImageCodec        = (*hfTextToImageCodec)(nil)
	_ SpeechRecognitionCodec  = (*hfSpeechRecognitionCodec)(nil)
)

// NewHuggingFaceProvider returns a HuggingFaceProvider composing every
// built-in per-task mixin. This is the recommended way to construct a
// HuggingFaceProvider.
func NewHuggingFaceProvider() HuggingFaceProvider {
	return HuggingFaceProvider{}
}

// ProviderSuffix returns an empty string. On OpenAI-compatible endpoints
// (e.g. chat completions), the HF router selects a provider server-side, so
// the Hugging Face provider appends no routing token to the model.
//
// A model-string suffix chooses how the router picks a provider: a provider
// name (e.g. "model:baseten") routes to that provider deterministically,
// while a routing policy (":fastest", ":cheapest", ":preferred") lets the
// router choose, with automatic failover to another provider. With no suffix
// the router applies the default ":fastest" policy. See the Inference
// Providers docs: https://huggingface.co/docs/inference-providers/main/en/index.
func (p HuggingFaceProvider) ProviderSuffix() string {
	return ""
}

// Name returns the name of the Hugging Face provider for debug purposes.
func (p HuggingFaceProvider) Name() string {
	return "huggingface"
}

// Protected structural request keys for the wire formats the Hugging Face
// codecs emit. ProviderParams entries merge into the body's top level after
// the typed request is marshaled, and these keys are re-applied afterwards so
// ProviderParams entries cannot override them: the resolved model is a
// routing decision, and inputs carries the request payload itself.
const (
	protectedChatModelKey     = "model"
	protectedPipelineInputKey = "inputs"
)

// hfParamsCodec is a [Codec] that wraps an inner codec and merges a request's
// ProviderParams into the wire body's top level. Entries override typed
// fields, matching the documented ProviderParams contract on every canonical
// request type. The ProviderParams map itself is treated as read-only.
//
// When the request carries no ProviderParams, the inner codec's output is
// returned byte-identical: the merge never changes bodies that need no
// merging. Otherwise the inner body is decoded into a JSON object map (with
// numeric fidelity preserved), the values of protectedKeys are captured,
// ProviderParams entries are merged in, the protected keys are re-applied, and
// the payload is re-marshaled. The inner codec's headers are passed through
// unchanged; Decode delegates to the inner codec.
type hfParamsCodec[Req, Resp any] struct {
	inner          Codec[Req, Resp]
	protectedKeys  []string
	providerParams func(Req) map[string]any
}

func (c hfParamsCodec[Req, Resp]) Encode(
	ctx context.Context,
	params EncodeParams[Req],
) (body []byte, headers http.Header, err error) {
	body, headers, err = c.inner.Encode(ctx, params)
	if err != nil {
		return nil, nil, err
	}

	providerParams := c.providerParams(params.Request)
	if len(providerParams) == 0 {
		return body, headers, nil
	}

	payload, err := decodeBodyToPayloadMap(body)
	if err != nil {
		return nil, nil, err
	}

	protected := make(map[string]any, len(c.protectedKeys))
	for _, key := range c.protectedKeys {
		if value, ok := payload[key]; ok {
			protected[key] = value
		}
	}
	mergeProviderParamsTopLevel(payload, providerParams)
	maps.Copy(payload, protected)

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
	}

	return body, headers, nil
}

func (c hfParamsCodec[Req, Resp]) Decode(ctx context.Context, params DecodeParams) (Resp, error) {
	return c.inner.Decode(ctx, params)
}

// requestProviderParams accessors are typed accessors handing each
// [hfParamsCodec] instantiation its request type's ProviderParams map. Go type
// parameters cannot access common struct fields, so each codec accessor passes
// the accessor for its concrete request type; a request type without the
// field fails to compile at its codec accessor instead of silently dropping
// the merge at runtime.
func chatRequestProviderParams(
	request hftypes.ChatRequest, //nolint:gocritic // hugeParam: taken by value so the accessor never mutates the caller's payload
) map[string]any {
	return request.ProviderParams
}

func featureExtractionRequestProviderParams(
	request hftypes.FeatureExtractionRequest,
) map[string]any {
	return request.ProviderParams
}

func featureExtractionBatchRequestProviderParams(
	request hftypes.FeatureExtractionBatchRequest,
) map[string]any {
	return request.ProviderParams
}

func fillMaskRequestProviderParams(request hftypes.FillMaskRequest) map[string]any {
	return request.ProviderParams
}

func imageClassificationRequestProviderParams(
	request hftypes.ImageClassificationRequest,
) map[string]any {
	return request.ProviderParams
}

func imageSegmentationRequestProviderParams(
	request hftypes.ImageSegmentationRequest,
) map[string]any {
	return request.ProviderParams
}

func objectDetectionRequestProviderParams(request hftypes.ObjectDetectionRequest) map[string]any {
	return request.ProviderParams
}

func questionAnsweringRequestProviderParams(
	request hftypes.QuestionAnsweringRequest,
) map[string]any {
	return request.ProviderParams
}

func speechRecognitionRequestProviderParams(
	request hftypes.SpeechRecognitionRequest,
) map[string]any {
	return request.ProviderParams
}

func summarizationRequestProviderParams(request hftypes.SummarizationRequest) map[string]any {
	return request.ProviderParams
}

func tableQuestionAnsweringRequestProviderParams(
	request hftypes.TableQuestionAnsweringRequest,
) map[string]any {
	return request.ProviderParams
}

func textClassificationRequestProviderParams(
	request hftypes.TextClassificationRequest,
) map[string]any {
	return request.ProviderParams
}

func textToImageRequestProviderParams(request hftypes.TextToImageRequest) map[string]any {
	return request.ProviderParams
}

func tokenClassificationRequestProviderParams(
	request hftypes.TokenClassificationRequest,
) map[string]any {
	return request.ProviderParams
}

func translationRequestProviderParams(request hftypes.TranslationRequest) map[string]any {
	return request.ProviderParams
}

func zeroShotTextClassificationRequestProviderParams(
	request hftypes.ZeroShotTextClassificationRequest,
) map[string]any {
	return request.ProviderParams
}

// hfChatParamsCodec wraps inner with the chat merge semantics: ProviderParams
// entries merge into the body's top level, with the resolved "model" key
// protected so entries cannot override model routing.
func hfChatParamsCodec[Req, Resp any](
	inner Codec[Req, Resp],
	providerParams func(Req) map[string]any,
) hfParamsCodec[Req, Resp] {
	return hfParamsCodec[Req, Resp]{
		inner:          inner,
		protectedKeys:  []string{protectedChatModelKey},
		providerParams: providerParams,
	}
}

// hfPipelineParamsCodec wraps inner with the pipeline merge semantics:
// ProviderParams entries merge into the body's top level, with the "inputs"
// key protected so entries cannot override the request payload itself.
func hfPipelineParamsCodec[Req, Resp any](
	inner Codec[Req, Resp],
	providerParams func(Req) map[string]any,
) hfParamsCodec[Req, Resp] {
	return hfParamsCodec[Req, Resp]{
		inner:          inner,
		protectedKeys:  []string{protectedPipelineInputKey},
		providerParams: providerParams,
	}
}

// HuggingFaceChatMixin provides the Hugging Face endpoint and codecs for chat
// completion, including streaming. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only chat.
//
// Its codecs merge a request's ProviderParams into the body's top level (the
// resolved "model" key is protected), so every provider embedding this mixin
// supports the documented ProviderParams escape hatch.
//
// Unary and streaming chat share the same endpoint, so embedding this mixin
// satisfies both ChatProvider and ChatStreamProvider.
type HuggingFaceChatMixin struct{}

// ChatEndpoint returns the endpoint for chat completion. Chat completions are
// served from a fixed OpenAI-compatible path, so this endpoint is
// model-independent: an empty params.Model is legal.
func (HuggingFaceChatMixin) ChatEndpoint(_ context.Context, _ EndpointParams) (Endpoint, error) {
	return Endpoint{Method: http.MethodPost, Path: "v1/chat/completions"}, nil
}

// ChatCodec returns the JSON codec for chat completion requests and
// responses, wrapped so the request's ProviderParams entries merge into the
// body's top level (with the resolved "model" key protected).
func (HuggingFaceChatMixin) ChatCodec() ChatCodec {
	return hfChatParamsCodec(
		JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
		chatRequestProviderParams,
	)
}

// ChatStreamCodec returns the JSON codec for streaming chat completion
// requests and responses, wrapped so the request's ProviderParams entries
// merge into the body's top level (with the resolved "model" key protected).
func (HuggingFaceChatMixin) ChatStreamCodec() ChatStreamCodec {
	return hfChatParamsCodec(
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		chatRequestProviderParams,
	)
}

// HuggingFaceFeatureExtractionMixin provides the Hugging Face endpoints and
// codecs for pooled and token-level feature extraction, including their batch
// variants. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only feature extraction.
//
// All feature-extraction variants share the same endpoints, so embedding this
// mixin satisfies every feature-extraction provider interface.
type HuggingFaceFeatureExtractionMixin struct{}

// FeatureExtractionBatchEndpoint returns the endpoint for batch feature extraction.
func (m HuggingFaceFeatureExtractionMixin) FeatureExtractionBatchEndpoint(
	ctx context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return m.FeatureExtractionEndpoint(ctx, params)
}

// FeatureExtractionEndpoint returns the endpoint for feature extraction.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "/pipeline/feature-extraction")
}

// FeatureExtractionPooledBatchCodec returns the codec for batch pooled
// feature-extraction requests and responses, wrapped so the request's
// ProviderParams entries merge into the body's top level (the "inputs" key is
// protected).
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionPooledBatchCodec() FeatureExtractionPooledBatchCodec {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionPooledEmbedding,
		]{},
		featureExtractionBatchRequestProviderParams,
	)
}

// FeatureExtractionPooledCodec returns the codec for pooled feature-extraction
// requests and responses, wrapped so the request's ProviderParams entries
// merge into the body's top level (the "inputs" key is protected).
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionPooledCodec() FeatureExtractionPooledCodec {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionPooledEmbedding,
		]{},
		featureExtractionRequestProviderParams,
	)
}

// FeatureExtractionTokenCodec returns the codec for token-level
// feature-extraction requests and responses, wrapped so the request's
// ProviderParams entries merge into the body's top level (the "inputs" key is
// protected).
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionTokenCodec() FeatureExtractionTokenCodec {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		featureExtractionRequestProviderParams,
	)
}

// FeatureExtractionTokenBatchCodec returns the codec for batch token-level
// feature-extraction requests and responses, wrapped so the request's
// ProviderParams entries merge into the body's top level (the "inputs" key is
// protected).
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionTokenBatchCodec() FeatureExtractionTokenBatchCodec {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		featureExtractionBatchRequestProviderParams,
	)
}

// HuggingFaceFillMaskMixin provides the Hugging Face endpoint and codec for
// fill-mask. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only fill-mask.
type HuggingFaceFillMaskMixin struct{}

// FillMaskEndpoint returns the endpoint for fill mask.
func (HuggingFaceFillMaskMixin) FillMaskEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// FillMaskCodec returns the JSON codec for fill-mask requests and responses,
// wrapped so the request's ProviderParams entries merge into the body's top
// level (the "inputs" key is protected).
func (HuggingFaceFillMaskMixin) FillMaskCodec() FillMaskCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{},
		fillMaskRequestProviderParams,
	)
}

// HuggingFaceImageClassificationMixin provides the Hugging Face endpoint and
// codec for image classification. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only image
// classification.
type HuggingFaceImageClassificationMixin struct{}

// ImageClassificationEndpoint returns the endpoint for image classification.
func (HuggingFaceImageClassificationMixin) ImageClassificationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// ImageClassificationCodec returns the codec for image-classification requests
// and responses, wrapped so the request's ProviderParams entries merge into
// the body's top level (the "inputs" key is protected).
func (HuggingFaceImageClassificationMixin) ImageClassificationCodec() ImageClassificationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.ImageClassificationRequest, []hftypes.ImageClassification]{},
		imageClassificationRequestProviderParams,
	)
}

// HuggingFaceImageSegmentationMixin provides the Hugging Face endpoint and
// codec for image segmentation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only image
// segmentation.
type HuggingFaceImageSegmentationMixin struct{}

// ImageSegmentationEndpoint returns the endpoint for image segmentation.
func (HuggingFaceImageSegmentationMixin) ImageSegmentationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// ImageSegmentationCodec returns the codec for image-segmentation requests and
// responses, wrapped so the request's ProviderParams entries merge into the
// body's top level (the "inputs" key is protected).
func (HuggingFaceImageSegmentationMixin) ImageSegmentationCodec() ImageSegmentationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]{},
		imageSegmentationRequestProviderParams,
	)
}

// HuggingFaceObjectDetectionMixin provides the Hugging Face endpoint and codec
// for object detection. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only object detection.
type HuggingFaceObjectDetectionMixin struct{}

// ObjectDetectionEndpoint returns the endpoint for object detection.
func (HuggingFaceObjectDetectionMixin) ObjectDetectionEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// ObjectDetectionCodec returns the codec for object-detection requests and
// responses, wrapped so the request's ProviderParams entries merge into the
// body's top level (the "inputs" key is protected).
func (HuggingFaceObjectDetectionMixin) ObjectDetectionCodec() ObjectDetectionCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{},
		objectDetectionRequestProviderParams,
	)
}

// HuggingFaceQuestionAnsweringMixin provides the Hugging Face endpoint and
// codec for question answering. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only question
// answering.
type HuggingFaceQuestionAnsweringMixin struct{}

// QuestionAnsweringEndpoint returns the endpoint for question answering.
func (HuggingFaceQuestionAnsweringMixin) QuestionAnsweringEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// QuestionAnsweringCodec returns the flexible question-answering codec, which
// accepts both single-object and array response shapes. Requests' ProviderParams
// entries merge into the body's top level (the "inputs" key is protected).
func (HuggingFaceQuestionAnsweringMixin) QuestionAnsweringCodec() QuestionAnsweringCodec {
	return hfQuestionAnsweringCodec{}
}

// HuggingFaceSpeechRecognitionMixin provides the Hugging Face endpoint and
// codec for speech recognition. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only speech
// recognition.
type HuggingFaceSpeechRecognitionMixin struct{}

// SpeechRecognitionEndpoint returns the endpoint for speech recognition.
func (HuggingFaceSpeechRecognitionMixin) SpeechRecognitionEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// SpeechRecognitionCodec returns the speech recognition codec, which encodes
// requests as raw bytes when neither parameters nor ProviderParams are present
// and as JSON (with base64-encoded input and top-level ProviderParams merging)
// otherwise, always decoding the JSON response.
func (HuggingFaceSpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	return hfSpeechRecognitionCodec{}
}

// HuggingFaceSummarizationMixin provides the Hugging Face endpoint and codec
// for summarization. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only summarization.
type HuggingFaceSummarizationMixin struct{}

// SummarizationEndpoint returns the endpoint for summarization.
func (HuggingFaceSummarizationMixin) SummarizationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// SummarizationCodec returns the codec for summarization requests and
// responses, wrapped so the request's ProviderParams entries merge into the
// body's top level (the "inputs" key is protected).
func (HuggingFaceSummarizationMixin) SummarizationCodec() SummarizationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{},
		summarizationRequestProviderParams,
	)
}

// HuggingFaceTableQuestionAnsweringMixin provides the Hugging Face endpoint and
// codec for table question answering. It is a mixin: embed it in a type that
// also implements Provider.Name and Provider.ProviderSuffix to serve only table
// question answering.
type HuggingFaceTableQuestionAnsweringMixin struct{}

// TableQuestionAnsweringEndpoint returns the endpoint for table question answering.
func (HuggingFaceTableQuestionAnsweringMixin) TableQuestionAnsweringEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// TableQuestionAnsweringCodec returns the codec for table question-answering
// requests and responses, wrapped so the request's ProviderParams entries
// merge into the body's top level (the "inputs" key is protected).
func (HuggingFaceTableQuestionAnsweringMixin) TableQuestionAnsweringCodec() TableQuestionAnsweringCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{},
		tableQuestionAnsweringRequestProviderParams,
	)
}

// HuggingFaceTextClassificationMixin provides the Hugging Face endpoint and
// codec for text classification. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only text
// classification.
type HuggingFaceTextClassificationMixin struct{}

// TextClassificationEndpoint returns the endpoint for text classification.
func (HuggingFaceTextClassificationMixin) TextClassificationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// TextClassificationCodec returns the flexible text-classification codec,
// which accepts both flat and single-element nested array response shapes.
// Requests' ProviderParams entries merge into the body's top level (the
// "inputs" key is protected).
func (HuggingFaceTextClassificationMixin) TextClassificationCodec() TextClassificationCodec {
	return hfTextClassificationCodec{}
}

// HuggingFaceTextToImageMixin provides the Hugging Face endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image.
type HuggingFaceTextToImageMixin struct{}

// TextToImageEndpoint returns the endpoint for text-to-image generation.
func (HuggingFaceTextToImageMixin) TextToImageEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// TextToImageCodec returns the text-to-image codec, which encodes requests as
// JSON and returns the generated image as raw bytes.
func (HuggingFaceTextToImageMixin) TextToImageCodec() TextToImageCodec {
	return hfTextToImageCodec{}
}

// HuggingFaceTokenClassificationMixin provides the Hugging Face endpoint and
// codec for token classification. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only token
// classification.
type HuggingFaceTokenClassificationMixin struct{}

// TokenClassificationEndpoint returns the endpoint for token classification.
func (HuggingFaceTokenClassificationMixin) TokenClassificationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// TokenClassificationCodec returns the codec for token-classification requests
// and responses, wrapped so the request's ProviderParams entries merge into
// the body's top level (the "inputs" key is protected).
func (HuggingFaceTokenClassificationMixin) TokenClassificationCodec() TokenClassificationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{},
		tokenClassificationRequestProviderParams,
	)
}

// HuggingFaceTranslationMixin provides the Hugging Face endpoint and codec for
// translation. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only translation.
type HuggingFaceTranslationMixin struct{}

// TranslationEndpoint returns the endpoint for translation.
func (HuggingFaceTranslationMixin) TranslationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// TranslationCodec returns the codec for translation requests and responses,
// wrapped so the request's ProviderParams entries merge into the body's top
// level (the "inputs" key is protected).
func (HuggingFaceTranslationMixin) TranslationCodec() TranslationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{},
		translationRequestProviderParams,
	)
}

// HuggingFaceZeroShotTextClassificationMixin provides the Hugging Face
// endpoint and codec for zero-shot text classification. It is a mixin: embed it
// in a type that also implements Provider.Name and Provider.ProviderSuffix to
// serve only zero-shot text classification.
type HuggingFaceZeroShotTextClassificationMixin struct{}

// ZeroShotTextClassificationEndpoint returns the endpoint for zero-shot text classification.
func (HuggingFaceZeroShotTextClassificationMixin) ZeroShotTextClassificationEndpoint(
	_ context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return hfModelEndpoint(params, "")
}

// ZeroShotTextClassificationCodec returns the codec for zero-shot
// text-classification requests and responses, wrapped so the request's
// ProviderParams entries merge into the body's top level (the "inputs" key is
// protected).
func (HuggingFaceZeroShotTextClassificationMixin) ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec {
	return hfPipelineParamsCodec(
		JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{},
		zeroShotTextClassificationRequestProviderParams,
	)
}

// hfQuestionAnsweringCodec is a flexible Codec for question answering that handles both single object and array responses.
// The API specification promises an array of objects as a response, but in reality there may only be a single object
// returned if top_k is not greater than 1.
type hfQuestionAnsweringCodec struct{}

func (hfQuestionAnsweringCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.QuestionAnsweringRequest],
) (body []byte, headers http.Header, err error) {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.QuestionAnsweringRequest,
			[]hftypes.QuestionAnswer,
		]{},
		questionAnsweringRequestProviderParams,
	).Encode(ctx, params)
}

func (hfQuestionAnsweringCodec) Decode(
	_ context.Context,
	params DecodeParams,
) ([]hftypes.QuestionAnswer, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, errUnexpectedContentType(contentType)
	}

	// Spec-compliant response
	var slice []hftypes.QuestionAnswer
	sliceErr := json.Unmarshal(params.Body, &slice)
	if sliceErr == nil {
		return slice, nil
	}

	// Observed response for some inputs with top_k unset or equal to 1.
	// Require the expected answer fields so unrelated objects are not
	// accepted as zero-valued answers.
	single, singleErr := decodeSingleQuestionAnswering(params.Body)
	if singleErr == nil {
		return []hftypes.QuestionAnswer{single}, nil
	}

	joinedErr := errors.Join(sliceErr, singleErr)
	if sdkErr := existingSDKError(joinedErr); sdkErr != nil {
		return nil, sdkErr
	}

	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: errFailedToDecodeResponseBody,
		Err:     joinedErr,
	}
}

func decodeSingleQuestionAnswering(data []byte) (hftypes.QuestionAnswer, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return hftypes.QuestionAnswer{}, err
	}
	answerRaw, answerPresent := fields["answer"]
	if !answerPresent {
		return hftypes.QuestionAnswer{}, errors.New(
			"single answer object missing answer",
		)
	}
	if isJSONNull(answerRaw) {
		return hftypes.QuestionAnswer{}, errors.New(
			"single answer object answer is null",
		)
	}
	scoreRaw, scorePresent := fields["score"]
	if !scorePresent {
		return hftypes.QuestionAnswer{}, errors.New(
			"single answer object missing score",
		)
	}
	if isJSONNull(scoreRaw) {
		return hftypes.QuestionAnswer{}, errors.New(
			"single answer object score is null",
		)
	}

	var answer hftypes.QuestionAnswer
	if err := json.Unmarshal(data, &answer); err != nil {
		return hftypes.QuestionAnswer{}, err
	}

	return answer, nil
}

func isJSONNull(raw json.RawMessage) bool {
	return strings.TrimSpace(string(raw)) == "null"
}

// existingSDKError returns the first *hferrors.SDKError in err's chain, if any.
func existingSDKError(err error) error {
	if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
		return sdkErr
	}

	return nil
}

// hfModelEndpoint returns a POST endpoint scoped to the given model. The suffix
// is appended to the model path and may be empty.
func hfModelEndpoint(params EndpointParams, suffix string) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{
		Method: http.MethodPost,
		Path:   hfModelPrefix + params.Model + suffix,
	}, nil
}

// hfTextClassificationCodec is a flexible Codec for text classification that
// handles both the documented flat response and the nested response returned by
// the hf-inference serverless API.
//
// The upstream API documentation declares a flat JSON array of classification
// objects for a single input:
//
//	[{"label": "positive", "score": 0.95}, ...]
//
// In reality the shape depends on how the model is served. The hf-inference
// serverless API returns a nested array: the outer array holds one entry per
// input and each inner array holds that input's classification objects, while
// dedicated Inference Endpoints return the documented flat array. The serverless
// API also accepts a batch of inputs (a list of strings) even though batch input
// is not part of the documented single-input schema, which is why results are
// grouped per input. A single input therefore arrives as a one-element outer
// array:
//
//	[[{"label": "positive", "score": 0.95}, ...]]
//
// A batch of N inputs yields an outer array of length N. The top_k parameter,
// when set, limits how many classification objects appear in each inner array.
// Decode accepts either shape and always returns the flat list of
// classifications the SDK promises for a single input.
type hfTextClassificationCodec struct{}

func (hfTextClassificationCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.TextClassificationRequest],
) (body []byte, headers http.Header, err error) {
	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.TextClassificationRequest,
			[]hftypes.TextClassification,
		]{},
		textClassificationRequestProviderParams,
	).Encode(ctx, params)
}

func (hfTextClassificationCodec) Decode(
	_ context.Context,
	params DecodeParams,
) ([]hftypes.TextClassification, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, errUnexpectedContentType(contentType)
	}

	// Documented/spec shape: a flat array of classification objects. Observed on
	// dedicated Inference Endpoints.
	var flat []hftypes.TextClassification
	flatErr := json.Unmarshal(params.Body, &flat)
	if flatErr == nil {
		return flat, nil
	}

	// Observed hf-inference serverless shape: a nested array with one inner array
	// per input. This single-input endpoint expects exactly one result set; more
	// than one is ambiguous and rejected.
	var nested [][]hftypes.TextClassification
	nestedErr := json.Unmarshal(params.Body, &nested)
	if nestedErr == nil {
		if len(nested) != 1 {
			return nil, &hferrors.SDKError{
				Kind: hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					"expected a single text-classification result set, got %d",
					len(nested),
				),
				Err: nil,
			}
		}

		return nested[0], nil
	}

	joinedErr := errors.Join(flatErr, nestedErr)
	if sdkErr := existingSDKError(joinedErr); sdkErr != nil {
		return nil, sdkErr
	}

	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: errFailedToDecodeResponseBody,
		Err:     joinedErr,
	}
}

// hfTextToImageCodec is a Codec for text-to-image that encodes canonical
// requests as JSON and decodes the generated image from the raw response body.
// The exact image subtype is provider-chosen, so requests advertise
// "Accept: */*" and content-type validation accepts any image media type,
// returning the body plus the parsed media type.
type hfTextToImageCodec struct{}

func (hfTextToImageCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	body, headers, err = hfPipelineParamsCodec(
		JSONCodec[
			hftypes.TextToImageRequest,
			hftypes.TextToImageResponse,
		]{},
		textToImageRequestProviderParams,
	).Encode(ctx, params)
	if err != nil {
		return nil, nil, err
	}
	headers.Set("Accept", mimeTypeAny)

	return body, headers, nil
}

func (hfTextToImageCodec) Decode(
	_ context.Context,
	params DecodeParams,
) (hftypes.TextToImageResponse, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, "image/") {
		return hftypes.TextToImageResponse{}, errUnexpectedMediaPrefix("image/", contentType)
	}

	return hftypes.TextToImageResponse{
		Image:            params.Body,
		MediaType:        mediaType,
		ProviderMetadata: nil,
	}, nil
}

// hfSpeechRecognitionCodec is a Codec for speech recognition that encodes
// requests as raw audio bytes when neither parameters nor ProviderParams are
// provided (matching the API requirement), and as JSON with base64-encoded
// input and top-level ProviderParams merging otherwise. Responses are always
// JSON.
type hfSpeechRecognitionCodec struct{}

func (hfSpeechRecognitionCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request
	if req.Parameters == nil && len(req.ProviderParams) == 0 {
		decoded, err := base64.StdEncoding.DecodeString(req.Input)
		if err != nil {
			return nil, nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: errFailedToDecodeBase64Input,
				Err:     err,
			}
		}

		headers = http.Header{}
		headers.Set("Content-Type", http.DetectContentType(decoded))
		headers.Set("Accept", "application/json")

		return decoded, headers, nil
	}

	return hfPipelineParamsCodec(
		JSONCodec[
			hftypes.SpeechRecognitionRequest,
			hftypes.SpeechRecognitionResponse,
		]{},
		speechRecognitionRequestProviderParams,
	).Encode(ctx, params)
}

func (hfSpeechRecognitionCodec) Decode(
	ctx context.Context,
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return JSONCodec[
		hftypes.SpeechRecognitionRequest,
		hftypes.SpeechRecognitionResponse,
	]{}.Decode(ctx, params)
}
