// Package providers defines the Provider interface and built-in provider implementations.
package providers

// Provider knows how to construct API endpoints for a given task and model.
type Provider interface {
	// Endpoint returns the API endpoint path for the given task and model.
	// It returns an error if the endpoint cannot be constructed.
	Endpoint(task, model string) (string, error)

	// ProviderSuffix returns the provider suffix appended to model IDs
	// for routing in OpenAI-compatible endpoints.
	ProviderSuffix() string
}
