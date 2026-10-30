package hfproviders

// zaiProviderName is the Z.ai provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":zai-org" provider pin.
const zaiProviderName = "zai-org"

// ZAIProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":zai-org" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Z.ai needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":zai-org" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Z.ai does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type ZAIProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that ZAIProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*ZAIProvider)(nil)
	_ ChatStreamProvider = (*ZAIProvider)(nil)
)

// NewZAIProvider returns a ZAIProvider that serves
// chat completion tasks.
func NewZAIProvider() ZAIProvider {
	return ZAIProvider{}
}

// ProviderSuffix returns the "zai-org" provider suffix for
// appending to the model selection in chat completion tasks.
func (p ZAIProvider) ProviderSuffix() string {
	return zaiProviderName
}

// Name returns the name of the Z.ai provider for debug purposes.
func (p ZAIProvider) Name() string {
	return zaiProviderName
}
