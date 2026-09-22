// Package providers defines the Provider interface and built-in provider implementations.
package providers

// Provider knows how to construct API endpoints and transform between HF-spec
// and provider-spec wire formats for a given task and model. Implementations
// must be safe for concurrent use.
type Provider interface {
	// Endpoint returns the API endpoint path for the given task and model.
	// It returns an error if the model is invalid or the task is unsupported.
	Endpoint(task Task, model string) (string, error)

	// ProviderSuffix returns the provider suffix appended to model IDs
	// for routing in OpenAI-compatible endpoints. An empty string means
	// no suffix is appended (e.g. for the default HuggingFace provider).
	ProviderSuffix() string

	// EncodeRequest transforms an HF-format request body into a provider-format
	// request body. hfContentType is the Content-Type of the HF-format body
	// (e.g. "application/json", "image/png"). For JSON tasks, hfBody is the
	// JSON serialization of the SDK's exported request type for the task;
	// for binary tasks it is the raw body. Returns the transformed body and
	// the Content-Type to use when sending to the provider.
	EncodeRequest(task Task, hfBody []byte, hfContentType string) (
		providerBody []byte, providerContentType string, err error,
	)

	// DecodeResponse transforms a provider-format response body into an
	// HF-format response body. providerContentType is the Content-Type of the
	// provider's response. The returned body must match the shape expected by
	// the SDK's exported response type for the task. Returns the transformed
	// body and the Content-Type of the transformed body.
	DecodeResponse(task Task, providerBody []byte, providerContentType string) (
		hfBody []byte, hfContentType string, err error,
	)
}
