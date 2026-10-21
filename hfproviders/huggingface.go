package hfproviders

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
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

// NewHuggingFaceProvider returns a HuggingFaceProvider with the default
// identity codec. This is the recommended way to construct a HuggingFaceProvider.
func NewHuggingFaceProvider() HuggingFaceProvider {
	return HuggingFaceProvider{}
}

// ProviderSuffix returns an empty string. On OpenAI-compatible endpoints
// (e.g. chat completions), the HF router selects a provider server-side, so
// the Hugging Face provider appends no routing pin to the model. A provider or
// selection policy can be pinned by appending a suffix to the model string
// (e.g. "model:sambanova", "model:fastest", "model:cheapest",
// "model:preferred"); see the Inference Providers docs for provider selection
// behavior:
// https://huggingface.co/docs/inference-providers/main/en/index.
func (p HuggingFaceProvider) ProviderSuffix() string {
	return ""
}

// Name returns the name of the Hugging Face provider for debug purposes.
func (p HuggingFaceProvider) Name() string {
	return "huggingface"
}

// HuggingFaceChatMixin provides the Hugging Face endpoint and codecs for chat
// completion, including streaming. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only chat.
//
// Unary and streaming chat share the same endpoint, so embedding this mixin
// satisfies both ChatProvider and ChatStreamProvider.
type HuggingFaceChatMixin struct{}

// ChatEndpoint returns the endpoint for chat completion. Chat completions are
// served from a fixed OpenAI-compatible path, so this endpoint is
// model-independent: an empty params.Model is legal.
func (HuggingFaceChatMixin) ChatEndpoint(_ EndpointParams) (Endpoint, error) {
	return Endpoint{Method: http.MethodPost, Path: "v1/chat/completions"}, nil
}

// ChatCodec returns the JSON codec for chat completion requests and responses.
func (HuggingFaceChatMixin) ChatCodec() ChatCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}
}

// ChatStreamCodec returns the JSON codec for streaming chat completion
// requests and responses.
func (HuggingFaceChatMixin) ChatStreamCodec() ChatStreamCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}
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
func (e HuggingFaceFeatureExtractionMixin) FeatureExtractionBatchEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return e.FeatureExtractionEndpoint(params)
}

// FeatureExtractionEndpoint returns the endpoint for feature extraction.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{
		Method: http.MethodPost,
		Path:   hfModelPrefix + params.Model + "/pipeline/feature-extraction",
	}, nil
}

// FeatureExtractionPooledBatchCodec returns the JSON codec for batch pooled
// feature-extraction requests and responses.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionPooledBatchCodec() FeatureExtractionPooledBatchCodec {
	return JSONCodec[
		hftypes.FeatureExtractionBatchRequest,
		[]hftypes.FeatureExtractionPooledEmbedding,
	]{}
}

// FeatureExtractionPooledCodec returns the JSON codec for pooled
// feature-extraction requests and responses.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionPooledCodec() FeatureExtractionPooledCodec {
	return JSONCodec[
		hftypes.FeatureExtractionRequest,
		hftypes.FeatureExtractionPooledEmbedding,
	]{}
}

// FeatureExtractionTokenCodec returns the JSON codec for token-level
// feature-extraction requests and responses.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionTokenCodec() FeatureExtractionTokenCodec {
	return JSONCodec[
		hftypes.FeatureExtractionRequest,
		hftypes.FeatureExtractionTokenEmbeddings,
	]{}
}

// FeatureExtractionTokenBatchCodec returns the JSON codec for batch
// token-level feature-extraction requests and responses.
func (HuggingFaceFeatureExtractionMixin) FeatureExtractionTokenBatchCodec() FeatureExtractionTokenBatchCodec {
	return JSONCodec[
		hftypes.FeatureExtractionBatchRequest,
		[]hftypes.FeatureExtractionTokenEmbeddings,
	]{}
}

// HuggingFaceFillMaskMixin provides the Hugging Face endpoint and codec for
// fill-mask. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only fill-mask.
type HuggingFaceFillMaskMixin struct{}

// FillMaskEndpoint returns the endpoint for fill mask.
func (HuggingFaceFillMaskMixin) FillMaskEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// FillMaskCodec returns the JSON codec for fill-mask requests and responses.
func (HuggingFaceFillMaskMixin) FillMaskCodec() FillMaskCodec {
	return JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{}
}

// HuggingFaceImageClassificationMixin provides the Hugging Face endpoint and
// codec for image classification. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only image
// classification.
type HuggingFaceImageClassificationMixin struct{}

// ImageClassificationEndpoint returns the endpoint for image classification.
func (HuggingFaceImageClassificationMixin) ImageClassificationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{
		Method: http.MethodPost,
		Path:   hfModelPrefix + params.Model,
	}, nil
}

// ImageClassificationCodec returns the JSON codec for image-classification
// requests and responses.
func (HuggingFaceImageClassificationMixin) ImageClassificationCodec() ImageClassificationCodec {
	return JSONCodec[hftypes.ImageClassificationRequest, []hftypes.ImageClassification]{}
}

// HuggingFaceImageSegmentationMixin provides the Hugging Face endpoint and
// codec for image segmentation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only image
// segmentation.
type HuggingFaceImageSegmentationMixin struct{}

// ImageSegmentationEndpoint returns the endpoint for image segmentation.
func (HuggingFaceImageSegmentationMixin) ImageSegmentationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// ImageSegmentationCodec returns the JSON codec for image-segmentation
// requests and responses.
func (HuggingFaceImageSegmentationMixin) ImageSegmentationCodec() ImageSegmentationCodec {
	return JSONCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]{}
}

// HuggingFaceObjectDetectionMixin provides the Hugging Face endpoint and codec
// for object detection. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only object detection.
type HuggingFaceObjectDetectionMixin struct{}

// ObjectDetectionEndpoint returns the endpoint for object detection.
func (HuggingFaceObjectDetectionMixin) ObjectDetectionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{
		Method: http.MethodPost,
		Path:   hfModelPrefix + params.Model,
	}, nil
}

// ObjectDetectionCodec returns the JSON codec for object-detection requests and responses.
func (HuggingFaceObjectDetectionMixin) ObjectDetectionCodec() ObjectDetectionCodec {
	return JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{}
}

// HuggingFaceQuestionAnsweringMixin provides the Hugging Face endpoint and
// codec for question answering. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only question
// answering.
type HuggingFaceQuestionAnsweringMixin struct{}

// QuestionAnsweringEndpoint returns the endpoint for question answering.
func (HuggingFaceQuestionAnsweringMixin) QuestionAnsweringEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// QuestionAnsweringCodec returns the flexible question-answering codec, which
// accepts both single-object and array response shapes.
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
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// SpeechRecognitionCodec returns the speech recognition codec, which encodes
// requests as raw bytes when no parameters are present and as JSON when
// parameters are provided, always decoding the JSON response.
func (HuggingFaceSpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	return hfSpeechRecognitionCodec{}
}

// HuggingFaceSummarizationMixin provides the Hugging Face endpoint and codec
// for summarization. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only summarization.
type HuggingFaceSummarizationMixin struct{}

// SummarizationEndpoint returns the endpoint for summarization.
func (HuggingFaceSummarizationMixin) SummarizationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// SummarizationCodec returns the JSON codec for summarization requests and
// responses.
func (HuggingFaceSummarizationMixin) SummarizationCodec() SummarizationCodec {
	return JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{}
}

// HuggingFaceTableQuestionAnsweringMixin provides the Hugging Face endpoint and
// codec for table question answering. It is a mixin: embed it in a type that
// also implements Provider.Name and Provider.ProviderSuffix to serve only table
// question answering.
type HuggingFaceTableQuestionAnsweringMixin struct{}

// TableQuestionAnsweringEndpoint returns the endpoint for table question answering.
func (HuggingFaceTableQuestionAnsweringMixin) TableQuestionAnsweringEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TableQuestionAnsweringCodec returns the JSON codec for table
// question-answering requests and responses.
func (HuggingFaceTableQuestionAnsweringMixin) TableQuestionAnsweringCodec() TableQuestionAnsweringCodec {
	return JSONCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{}
}

// HuggingFaceTextClassificationMixin provides the Hugging Face endpoint and
// codec for text classification. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only text
// classification.
type HuggingFaceTextClassificationMixin struct{}

// TextClassificationEndpoint returns the endpoint for text classification.
func (HuggingFaceTextClassificationMixin) TextClassificationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TextClassificationCodec returns the flexible text-classification codec,
// which accepts both flat and single-element nested array response shapes.
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
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
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
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TokenClassificationCodec returns the JSON codec for token-classification
// requests and responses.
func (HuggingFaceTokenClassificationMixin) TokenClassificationCodec() TokenClassificationCodec {
	return JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{}
}

// HuggingFaceTranslationMixin provides the Hugging Face endpoint and codec for
// translation. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only translation.
type HuggingFaceTranslationMixin struct{}

// TranslationEndpoint returns the endpoint for translation.
func (HuggingFaceTranslationMixin) TranslationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TranslationCodec returns the JSON codec for translation requests and
// responses.
func (HuggingFaceTranslationMixin) TranslationCodec() TranslationCodec {
	return JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{}
}

// HuggingFaceZeroShotTextClassificationMixin provides the Hugging Face
// endpoint and codec for zero-shot text classification. It is a mixin: embed it
// in a type that also implements Provider.Name and Provider.ProviderSuffix to
// serve only zero-shot text classification.
type HuggingFaceZeroShotTextClassificationMixin struct{}

// ZeroShotTextClassificationEndpoint returns the endpoint for zero-shot text classification.
func (HuggingFaceZeroShotTextClassificationMixin) ZeroShotTextClassificationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// ZeroShotTextClassificationCodec returns the JSON codec for zero-shot
// text-classification requests and responses.
func (HuggingFaceZeroShotTextClassificationMixin) ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec {
	return JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{}
}

// hfQuestionAnsweringCodec is a flexible Codec for question answering that handles both single object and array responses.
// The API specification promises an array of objects as a response, but in reality there may only be a single object
// returned if top_k is not greater than 1.
type hfQuestionAnsweringCodec struct{}

func (hfQuestionAnsweringCodec) Encode(
	params EncodeParams[hftypes.QuestionAnsweringRequest],
) (body []byte, headers http.Header, err error) {
	return JSONCodec[
		hftypes.QuestionAnsweringRequest,
		[]hftypes.QuestionAnswer,
	]{}.Encode(params)
}

func (hfQuestionAnsweringCodec) Decode(params DecodeParams) ([]hftypes.QuestionAnswer, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
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
	params EncodeParams[hftypes.TextClassificationRequest],
) (body []byte, headers http.Header, err error) {
	return JSONCodec[
		hftypes.TextClassificationRequest,
		[]hftypes.TextClassification,
	]{}.Encode(params)
}

func (hfTextClassificationCodec) Decode(params DecodeParams) ([]hftypes.TextClassification, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
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
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	body, headers, err = JSONCodec[
		hftypes.TextToImageRequest,
		hftypes.TextToImageResponse,
	]{}.Encode(params)
	if err != nil {
		return nil, nil, err
	}
	headers.Set("Accept", "*/*")

	return body, headers, nil
}

func (hfTextToImageCodec) Decode(
	params DecodeParams,
) (hftypes.TextToImageResponse, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, "image/") {
		return hftypes.TextToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected an image Content-Type, got " + contentType,
			Err:     nil,
		}
	}

	return hftypes.TextToImageResponse{
		Image:     params.Body,
		MediaType: mediaType,
	}, nil
}

// hfSpeechRecognitionCodec is a Codec for speech recognition that encodes
// requests as raw audio bytes when no parameters are provided (matching the
// API requirement), and as JSON with base64-encoded input when parameters
// are present. Responses are always JSON.
type hfSpeechRecognitionCodec struct{}

func (hfSpeechRecognitionCodec) Encode(
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request
	if req.Parameters == nil {
		decoded, err := base64.StdEncoding.DecodeString(req.Input)
		if err != nil {
			return nil, nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "failed to decode base64 input",
				Err:     err,
			}
		}

		headers = http.Header{}
		headers.Set("Content-Type", http.DetectContentType(decoded))
		headers.Set("Accept", "application/json")

		return decoded, headers, nil
	}

	return JSONCodec[
		hftypes.SpeechRecognitionRequest,
		hftypes.SpeechRecognitionResponse,
	]{}.Encode(params)
}

func (hfSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return JSONCodec[
		hftypes.SpeechRecognitionRequest,
		hftypes.SpeechRecognitionResponse,
	]{}.Decode(params)
}
