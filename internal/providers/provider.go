// Package providers defines the Provider interface and built-in provider implementations.
package providers

// Provider knows how to construct API endpoints for a given task and model.
// Implementations must be safe for concurrent use.
type Provider interface {
	// Endpoint returns the API endpoint path for the given task and model.
	// It returns an error if the model is invalid or the task is unsupported.
	Endpoint(task Task, model string) (string, error)
	// ProviderSuffix returns the provider suffix appended to model IDs
	// for routing in OpenAI-compatible endpoints. An empty string means
	// no suffix is appended (e.g. for the default HuggingFace provider).
	ProviderSuffix() string
}
