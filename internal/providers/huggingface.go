package providers

import (
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
)

// HuggingFaceProvider implements Provider for the HuggingFace inference API.
type HuggingFaceProvider struct{}

// ProviderSuffix returns the provider suffix appended to model IDs
// for routing in OpenAI-compatible endpoints.
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

	switch task {
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
