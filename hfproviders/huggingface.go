package hfproviders

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// HuggingFaceProvider implements Provider for the HuggingFace inference API.
// It embeds DefaultCodec, inheriting the identity request and response
// transforms: the HuggingFace API is the reference wire format, so no
// transformation is needed.
type HuggingFaceProvider struct{}

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

// Endpoint returns the API endpoint path for the given task and model.
// It returns an error if model is empty or the task is unsupported.
//
// The endpoint depends on the task: pipeline tasks (e.g. feature-extraction,
// sentence-similarity) return a task-specific path, chat-completion returns
// the OpenAI-compatible chat completions path, and all other tasks return the
// default model path.
func (p HuggingFaceProvider) Endpoint(task Task, model string) (string, error) {
	if model == "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "model is required",
			Err:     nil,
		}
	}

	switch task { //nolint:exhaustive // Additional hf-inference tasks are not yet supported by either the SDK or upstream API.
	case TaskFeatureExtraction, TaskSentenceSimilarity:
		return "hf-inference/models/" + model + "/pipeline/" + string(task), nil
	case TaskChatCompletion:
		return "v1/chat/completions", nil
	case TaskTextClassification, TaskZeroShotTextClassification, TaskTokenClassification,
		TaskQuestionAnswering, TaskTableQuestionAnswering, TaskFillMask,
		TaskSummarization, TaskTranslation:
		return "hf-inference/models/" + model, nil
	default:
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "unsupported task " + string(task),
			Err:     nil,
		}
	}
}

// ChatCodec returns the JSON codec for chat completion requests and responses.
func (HuggingFaceProvider) ChatCodec() ChatCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}
}

// ChatStreamCodec returns the JSON codec for streaming chat completion
// requests and responses.
func (HuggingFaceProvider) ChatStreamCodec() ChatStreamCodec {
	return JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}
}

// FeatureExtractionBatchCodec returns the JSON codec for batch
// feature-extraction requests and responses.
func (HuggingFaceProvider) FeatureExtractionBatchCodec() FeatureExtractionBatchCodec {
	return JSONCodec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]{}
}

// FeatureExtractionCodec returns the JSON codec for feature-extraction
// requests and responses.
func (HuggingFaceProvider) FeatureExtractionCodec() FeatureExtractionCodec {
	return JSONCodec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]{}
}

// FillMaskBatchCodec returns the JSON codec for batch fill-mask requests
// and responses.
func (HuggingFaceProvider) FillMaskBatchCodec() FillMaskBatchCodec {
	return JSONCodec[hftypes.FillMaskBatchRequest, [][]hftypes.FillMaskPrediction]{}
}

// FillMaskCodec returns the JSON codec for fill-mask requests and responses.
func (HuggingFaceProvider) FillMaskCodec() FillMaskCodec {
	return JSONCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{}
}

// QuestionAnsweringCodec returns the JSON codec for question-answering
// requests and responses.
func (HuggingFaceProvider) QuestionAnsweringCodec() QuestionAnsweringCodec {
	return JSONCodec[hftypes.QuestionAnsweringRequest, []hftypes.QuestionAnswering]{}
}

// SummarizationBatchCodec returns the JSON codec for batch summarization
// requests and responses.
func (HuggingFaceProvider) SummarizationBatchCodec() SummarizationBatchCodec {
	return JSONCodec[hftypes.SummarizationBatchRequest, []hftypes.Summarization]{}
}

// SummarizationCodec returns the JSON codec for summarization requests and
// responses.
func (HuggingFaceProvider) SummarizationCodec() SummarizationCodec {
	return JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{}
}

// TableQuestionAnsweringCodec returns the JSON codec for table
// question-answering requests and responses.
func (HuggingFaceProvider) TableQuestionAnsweringCodec() TableQuestionAnsweringCodec {
	return JSONCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{}
}

// TextClassificationBatchCodec returns the JSON codec for batch
// text-classification requests and responses.
func (HuggingFaceProvider) TextClassificationBatchCodec() TextClassificationBatchCodec {
	return JSONCodec[hftypes.TextClassificationBatchRequest, [][]hftypes.TextClassification]{}
}

// TextClassificationCodec returns the JSON codec for text-classification
// requests and responses.
func (HuggingFaceProvider) TextClassificationCodec() TextClassificationCodec {
	return JSONCodec[hftypes.TextClassificationRequest, []hftypes.TextClassification]{}
}

// TokenClassificationBatchCodec returns the JSON codec for batch
// token-classification requests and responses.
func (HuggingFaceProvider) TokenClassificationBatchCodec() TokenClassificationBatchCodec {
	return JSONCodec[hftypes.TokenClassificationBatchRequest, [][]hftypes.TokenClassification]{}
}

// TokenClassificationCodec returns the JSON codec for token-classification
// requests and responses.
func (HuggingFaceProvider) TokenClassificationCodec() TokenClassificationCodec {
	return JSONCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{}
}

// TranslationBatchCodec returns the JSON codec for batch translation
// requests and responses.
func (HuggingFaceProvider) TranslationBatchCodec() TranslationBatchCodec {
	return JSONCodec[hftypes.TranslationBatchRequest, []hftypes.Translation]{}
}

// TranslationCodec returns the JSON codec for translation requests and
// responses.
func (HuggingFaceProvider) TranslationCodec() TranslationCodec {
	return JSONCodec[hftypes.TranslationRequest, []hftypes.Translation]{}
}

// ZeroShotTextClassificationBatchCodec returns the JSON codec for batch
// zero-shot text-classification requests and responses.
func (HuggingFaceProvider) ZeroShotTextClassificationBatchCodec() ZeroShotTextClassificationBatchCodec {
	return JSONCodec[hftypes.ZeroShotTextClassificationBatchRequest, [][]hftypes.ZeroShotTextClassification]{}
}

// ZeroShotTextClassificationCodec returns the JSON codec for zero-shot
// text-classification requests and responses.
func (HuggingFaceProvider) ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec {
	return JSONCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{}
}
