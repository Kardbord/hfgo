package hfproviders

// fireworksAIProviderName is the Fireworks AI provider slug and the
// model-suffix token appended to chat model IDs, matching the router's
// ":fireworks-ai" provider pin.
const fireworksAIProviderName = "fireworks-ai"

// FireworksAIProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":fireworks-ai" provider suffix to the model ID.
//
// Unlike [WaveSpeedProvider], Fireworks needs no [ModelResolver]: the chat
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
type FireworksAIProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that FireworksAIProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*FireworksAIProvider)(nil)
	_ ChatStreamProvider = (*FireworksAIProvider)(nil)
)

// NewFireworksAIProvider returns a FireworksAIProvider that serves
// chat completion tasks.
func NewFireworksAIProvider() FireworksAIProvider {
	return FireworksAIProvider{}
}

// ProviderSuffix returns the "fireworks-ai" provider suffix for
// appending to the model selection in chat completion tasks.
func (p FireworksAIProvider) ProviderSuffix() string {
	return fireworksAIProviderName
}

// Name returns the name of the Fireworks AI provider for debug purposes.
func (p FireworksAIProvider) Name() string {
	return fireworksAIProviderName
}
