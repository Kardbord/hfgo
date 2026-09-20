package providers

import (
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
)

// HuggingFaceProvider implements Provider for the HuggingFace inference API.
// It embeds DefaultCodec, inheriting the identity request and response
// transforms: the HuggingFace API is the reference wire format, so no
// transformation is needed.
type HuggingFaceProvider struct {
	DefaultCodec
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
