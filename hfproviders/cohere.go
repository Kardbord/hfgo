package hfproviders

// cohereProviderName is the Cohere provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":cohere" provider pin.
const cohereProviderName = "cohere"

// CohereProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":cohere" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Cohere needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":cohere" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Cohere does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type CohereProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that CohereProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*CohereProvider)(nil)
	_ ChatStreamProvider = (*CohereProvider)(nil)
)

// NewCohereProvider returns a CohereProvider that serves
// chat completion tasks.
func NewCohereProvider() CohereProvider {
	return CohereProvider{}
}

// ProviderSuffix returns the "cohere" provider suffix for
// appending to the model selection in chat completion tasks.
func (p CohereProvider) ProviderSuffix() string {
	return cohereProviderName
}

// Name returns the name of the Cohere provider for debug purposes.
func (p CohereProvider) Name() string {
	return cohereProviderName
}
