package hfproviders

// fireworksProviderName is the Fireworks provider slug and the model-suffix
// token appended to chat model IDs, matching the router's ":fireworks-ai"
// provider pin.
const fireworksProviderName = "fireworks-ai"

// FireworksProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":fireworks-ai" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Fireworks needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":fireworks-ai" suffix is a deterministic provider pin —
// the router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Fireworks does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type FireworksProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that FireworksProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*FireworksProvider)(nil)
	_ ChatStreamProvider = (*FireworksProvider)(nil)
)

// NewFireworksProvider returns a FireworksProvider that serves
// chat completion tasks.
func NewFireworksProvider() FireworksProvider {
	return FireworksProvider{}
}

// ProviderSuffix returns the "fireworks-ai" provider suffix for
// appending to the model selection in chat completion tasks.
func (p FireworksProvider) ProviderSuffix() string {
	return fireworksProviderName
}

// Name returns the name of the Fireworks provider for debug purposes.
func (p FireworksProvider) Name() string {
	return fireworksProviderName
}
