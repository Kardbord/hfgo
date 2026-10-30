package hfproviders

// publicAIProviderName is the Public AI provider slug and the model-suffix
// token appended to chat model IDs, matching the router's ":publicai"
// provider pin.
const publicAIProviderName = "publicai"

// PublicAIProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":publicai" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Public AI needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":publicai" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Public AI does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type PublicAIProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that PublicAIProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*PublicAIProvider)(nil)
	_ ChatStreamProvider = (*PublicAIProvider)(nil)
)

// NewPublicAIProvider returns a PublicAIProvider that serves
// chat completion tasks.
func NewPublicAIProvider() PublicAIProvider {
	return PublicAIProvider{}
}

// ProviderSuffix returns the "publicai" provider suffix for
// appending to the model selection in chat completion tasks.
func (p PublicAIProvider) ProviderSuffix() string {
	return publicAIProviderName
}

// Name returns the name of the Public AI provider for debug purposes.
func (p PublicAIProvider) Name() string {
	return publicAIProviderName
}
