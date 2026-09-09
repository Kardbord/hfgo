package providers

import "errors"

// HuggingFaceProvider implements Provider for the HuggingFace inference API.
type HuggingFaceProvider struct{}

// ProviderSuffix returns the provider suffix appended to model IDs
// for routing in OpenAI-compatible endpoints.
func (p HuggingFaceProvider) ProviderSuffix() string {
	return ""
}

// Endpoint returns the API endpoint path for the given task and model.
// It returns an error if model is empty.
//
// The endpoint depends on the task: pipeline tasks (e.g. "feature-extraction",
// "sentence-similarity") return a task-specific path, "chat-completion" returns
// the OpenAI-compatible chat completions path, and all other tasks return the
// default model path.
func (p HuggingFaceProvider) Endpoint(task, model string) (string, error) {
	if model == "" {
		return "", errors.New("model is required")
	}

	switch task {
	case "feature-extraction", "sentence-similarity":
		return "hf-inference/models/" + model + "/pipeline/" + task, nil
	case "chat-completion":
		return "/v1/chat/completions", nil
	default:
		return "hf-inference/models/" + model, nil
	}
}
