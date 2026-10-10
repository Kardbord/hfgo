package hfproviders

import (
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

// HuggingFaceProvider implements Provider for the HuggingFace inference API.
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
// the HuggingFace provider appends no routing pin to the model. A provider or
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

// HuggingFaceCodecs provides default HuggingFace Codec implementations
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

// DetectObjectsCodec returns the JSON codec for object-detection requests and responses.
func (HuggingFaceCodecs) DetectObjectsCodec() DetectObjectsCodec {
	return JSONCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{}
}

// FeatureExtractionBatchCodec returns the JSON codec for batch
// feature-extraction requests and responses.
func (HuggingFaceCodecs) FeatureExtractionBatchCodec() FeatureExtractionBatchCodec {
	return JSONCodec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]{}
}

// FeatureExtractionCodec returns the JSON codec for feature-extraction
// requests and responses.
func (HuggingFaceCodecs) FeatureExtractionCodec() FeatureExtractionCodec {
	return JSONCodec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]{}
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

// hfQuestionAnsweringCodec is a flexible Codec for question answering that handles both single object and array responses.
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

	var slice []hftypes.QuestionAnswering
	sliceErr := json.Unmarshal(params.Body, &slice)
	if sliceErr == nil {
		return slice, nil
	}

	var single hftypes.QuestionAnswering
	singleErr := json.Unmarshal(params.Body, &single)
	if singleErr == nil {
		return []hftypes.QuestionAnswering{single}, nil
	}

	return nil, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: errFailedToDecodeResponseBody,
		Err:     errors.Join(sliceErr, singleErr),
	}
}

// hfTextClassificationCodec is a flexible Codec for text classification that handles both flat and nested array responses.
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

	var flat []hftypes.TextClassification
	flatErr := json.Unmarshal(params.Body, &flat)
	if flatErr == nil {
		return flat, nil
	}

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

// HuggingFaceEndpoints provides default HuggingFace endpoint implementations
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

// DetectObjectsEndpoint returns the endpoint for object detection.
func (HuggingFaceEndpoints) DetectObjectsEndpoint(params EndpointParams) (Endpoint, error) {
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
