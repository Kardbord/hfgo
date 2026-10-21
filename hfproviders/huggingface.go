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
type HuggingFaceProvider struct {
	HuggingFaceEndpoints
	HuggingFaceCodecs
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

// HuggingFaceCodecs provides default Hugging Face Codec implementations
// that custom providers or test mocks can embed.
//
// Embed it only if you support every task it covers: embedding hands you
// codec methods for tasks you may not intend to serve. Providers supporting
// a subset of tasks should implement just the relevant per-task codec
// methods instead.
type HuggingFaceCodecs struct{}

// ChatCodec returns the JSON codec for chat completion requests and responses.
func (HuggingFaceCodecs) ChatCodec() ChatCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}
}

// ChatStreamCodec returns the JSON codec for streaming chat completion
// requests and responses.
func (HuggingFaceCodecs) ChatStreamCodec() ChatStreamCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}
}

// ObjectDetectionCodec returns the JSON codec for object-detection requests and responses.
func (HuggingFaceCodecs) ObjectDetectionCodec() ObjectDetectionCodec {
	return JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{}
}

// FeatureExtractionBatchCodec returns the JSON codec for batch
// feature-extraction requests and responses.
func (HuggingFaceCodecs) FeatureExtractionBatchCodec() FeatureExtractionBatchCodec {
	return JSONCodec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]{}
}

// FeatureExtractionCodec returns the JSON codec for feature-extraction
// requests and pooled responses.
func (HuggingFaceCodecs) FeatureExtractionCodec() FeatureExtractionCodec {
	return JSONCodec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]{}
}

// FeatureExtractionTokenCodec returns the JSON codec for feature-extraction
// requests and token-level responses.
func (HuggingFaceCodecs) FeatureExtractionTokenCodec() FeatureExtractionTokenCodec {
	return JSONCodec[
		hftypes.FeatureExtractionRequest,
		hftypes.FeatureExtractionTokens,
	]{}
}

// FeatureExtractionBatchTokenCodec returns the JSON codec for batch
// feature-extraction requests and token-level responses.
func (HuggingFaceCodecs) FeatureExtractionBatchTokenCodec() FeatureExtractionBatchTokenCodec {
	return JSONCodec[
		hftypes.FeatureExtractionBatchRequest,
		hftypes.FeatureExtractionBatchTokens,
	]{}
}

// FillMaskCodec returns the JSON codec for fill-mask requests and responses.
func (HuggingFaceCodecs) FillMaskCodec() FillMaskCodec {
	return JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{}
}

// ImageClassificationCodec returns the JSON codec for image-classification
// requests and responses.
func (HuggingFaceCodecs) ImageClassificationCodec() ImageClassificationCodec {
	return JSONCodec[hftypes.ImageClassificationRequest, []hftypes.ImageClassification]{}
}

// ImageSegmentationCodec returns the JSON codec for image-segmentation
// requests and responses.
func (HuggingFaceCodecs) ImageSegmentationCodec() ImageSegmentationCodec {
	return JSONCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]{}
}

// QuestionAnsweringCodec returns the flexible question-answering codec, which
// accepts both single-object and array response shapes.
func (HuggingFaceCodecs) QuestionAnsweringCodec() QuestionAnsweringCodec {
	return hfQuestionAnsweringCodec{}
}

// SummarizationCodec returns the JSON codec for summarization requests and
// responses.
func (HuggingFaceCodecs) SummarizationCodec() SummarizationCodec {
	return JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{}
}

// TableQuestionAnsweringCodec returns the JSON codec for table
// question-answering requests and responses.
func (HuggingFaceCodecs) TableQuestionAnsweringCodec() TableQuestionAnsweringCodec {
	return JSONCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{}
}

// TextClassificationCodec returns the flexible text-classification codec,
// which accepts both flat and single-element nested array response shapes.
func (HuggingFaceCodecs) TextClassificationCodec() TextClassificationCodec {
	return hfTextClassificationCodec{}
}

// TextToImageCodec returns the text-to-image codec, which encodes requests as
// JSON and returns the generated image as raw bytes.
func (HuggingFaceCodecs) TextToImageCodec() TextToImageCodec {
	return hfTextToImageCodec{}
}

// TokenClassificationCodec returns the JSON codec for token-classification
// requests and responses.
func (HuggingFaceCodecs) TokenClassificationCodec() TokenClassificationCodec {
	return JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{}
}

// TranslationCodec returns the JSON codec for translation requests and
// responses.
func (HuggingFaceCodecs) TranslationCodec() TranslationCodec {
	return JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{}
}

// ZeroShotTextClassificationCodec returns the JSON codec for zero-shot
// text-classification requests and responses.
func (HuggingFaceCodecs) ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec {
	return JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{}
}

// SpeechRecognitionCodec returns the speech recognition codec, which encodes
// requests as raw bytes when no parameters are present and as JSON when
// parameters are provided, always decoding the JSON response.
func (HuggingFaceCodecs) SpeechRecognitionCodec() SpeechRecognitionCodec {
	return hfSpeechRecognitionCodec{}
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
		[]hftypes.QuestionAnswering,
	]{}.Encode(params)
}

func (hfQuestionAnsweringCodec) Decode(params DecodeParams) ([]hftypes.QuestionAnswering, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
	}

	// Spec-compliant response
	var slice []hftypes.QuestionAnswering
	sliceErr := json.Unmarshal(params.Body, &slice)
	if sliceErr == nil {
		return slice, nil
	}

	// Observed response for some inputs with top_k unset or equal to 1.
	// Require the expected answer fields so unrelated objects are not
	// accepted as zero-valued answers.
	single, singleErr := decodeSingleQuestionAnswering(params.Body)
	if singleErr == nil {
		return []hftypes.QuestionAnswering{single}, nil
	}

	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: errFailedToDecodeResponseBody,
		Err:     errors.Join(sliceErr, singleErr),
	}
}

func decodeSingleQuestionAnswering(data []byte) (hftypes.QuestionAnswering, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return hftypes.QuestionAnswering{}, err
	}
	answerRaw, answerPresent := fields["answer"]
	if !answerPresent {
		return hftypes.QuestionAnswering{}, errors.New(
			"single answer object missing answer",
		)
	}
	if isJSONNull(answerRaw) {
		return hftypes.QuestionAnswering{}, errors.New(
			"single answer object answer is null",
		)
	}
	scoreRaw, scorePresent := fields["score"]
	if !scorePresent {
		return hftypes.QuestionAnswering{}, errors.New(
			"single answer object missing score",
		)
	}
	if isJSONNull(scoreRaw) {
		return hftypes.QuestionAnswering{}, errors.New(
			"single answer object score is null",
		)
	}

	var answer hftypes.QuestionAnswering
	if err := json.Unmarshal(data, &answer); err != nil {
		return hftypes.QuestionAnswering{}, err
	}

	return answer, nil
}

func isJSONNull(raw json.RawMessage) bool {
	return strings.TrimSpace(string(raw)) == "null"
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

	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: errFailedToDecodeResponseBody,
		Err:     errors.Join(flatErr, nestedErr),
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

// HuggingFaceEndpoints provides default Hugging Face endpoint implementations
// that custom providers or test mocks can embed.
//
// Embed it only if you support every task it covers: embedding hands you
// endpoint methods for tasks you may not intend to serve. Providers
// supporting a subset of tasks should implement just the relevant per-task
// endpoint methods instead.
type HuggingFaceEndpoints struct{}

// ChatEndpoint returns the endpoint for chat completion. Chat completions are
// served from a fixed OpenAI-compatible path, so this endpoint is
// model-independent: an empty params.Model is legal.
func (HuggingFaceEndpoints) ChatEndpoint(_ EndpointParams) (Endpoint, error) {
	return Endpoint{Method: http.MethodPost, Path: "v1/chat/completions"}, nil
}

// ObjectDetectionEndpoint returns the endpoint for object detection.
func (HuggingFaceEndpoints) ObjectDetectionEndpoint(params EndpointParams) (Endpoint, error) {
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

// ImageClassificationEndpoint returns the endpoint for image classification.
func (HuggingFaceEndpoints) ImageClassificationEndpoint(
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

// FeatureExtractionBatchEndpoint returns the endpoint for batch feature extraction.
func (e HuggingFaceEndpoints) FeatureExtractionBatchEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return e.FeatureExtractionEndpoint(params)
}

// FeatureExtractionEndpoint returns the endpoint for feature extraction.
func (HuggingFaceEndpoints) FeatureExtractionEndpoint(params EndpointParams) (Endpoint, error) {
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

// ImageSegmentationEndpoint returns the endpoint for image segmentation.
func (HuggingFaceEndpoints) ImageSegmentationEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TextClassificationEndpoint returns the endpoint for text classification.
func (HuggingFaceEndpoints) TextClassificationEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// ZeroShotTextClassificationEndpoint returns the endpoint for zero-shot text classification.
func (HuggingFaceEndpoints) ZeroShotTextClassificationEndpoint(
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

// TokenClassificationEndpoint returns the endpoint for token classification.
func (HuggingFaceEndpoints) TokenClassificationEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// QuestionAnsweringEndpoint returns the endpoint for question answering.
func (HuggingFaceEndpoints) QuestionAnsweringEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TableQuestionAnsweringEndpoint returns the endpoint for table question answering.
func (HuggingFaceEndpoints) TableQuestionAnsweringEndpoint(
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

// TextToImageEndpoint returns the endpoint for text-to-image generation.
func (HuggingFaceEndpoints) TextToImageEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// FillMaskEndpoint returns the endpoint for fill mask.
func (HuggingFaceEndpoints) FillMaskEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// SummarizationEndpoint returns the endpoint for summarization.
func (HuggingFaceEndpoints) SummarizationEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// TranslationEndpoint returns the endpoint for translation.
func (HuggingFaceEndpoints) TranslationEndpoint(params EndpointParams) (Endpoint, error) {
	if params.Model == "" {
		return Endpoint{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: errModelIsRequired,
			Err:     nil,
		}
	}

	return Endpoint{Method: http.MethodPost, Path: hfModelPrefix + params.Model}, nil
}

// SpeechRecognitionEndpoint returns the endpoint for speech recognition.
func (HuggingFaceEndpoints) SpeechRecognitionEndpoint(
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
