package hfproviders

// basetenProviderName is the Baseten provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":baseten" provider pin.
const basetenProviderName = "baseten"

// BasetenProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":baseten" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Baseten needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":baseten" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Baseten does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type BasetenProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that BasetenProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*BasetenProvider)(nil)
	_ ChatStreamProvider = (*BasetenProvider)(nil)
)

// NewBasetenProvider returns a BasetenProvider that serves
// chat completion tasks.
func NewBasetenProvider() BasetenProvider {
	return BasetenProvider{}
}

// ProviderSuffix returns the "baseten" provider suffix for
// appending to the model selection in chat completion tasks.
func (p BasetenProvider) ProviderSuffix() string {
	return basetenProviderName
}

// Name returns the name of the Baseten provider for debug purposes.
func (p BasetenProvider) Name() string {
	return basetenProviderName
}
