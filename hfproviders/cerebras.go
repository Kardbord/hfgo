package hfproviders

// cerebrasProviderName is the Cerebras provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":cerebras" provider pin.
const cerebrasProviderName = "cerebras"

// CerebrasProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":cerebras" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Cerebras needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":cerebras" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Cerebras does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type CerebrasProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that CerebrasProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*CerebrasProvider)(nil)
	_ ChatStreamProvider = (*CerebrasProvider)(nil)
)

// NewCerebrasProvider returns a CerebrasProvider that serves
// chat completion tasks.
func NewCerebrasProvider() CerebrasProvider {
	return CerebrasProvider{}
}

// ProviderSuffix returns the "cerebras" provider suffix for
// appending to the model selection in chat completion tasks.
func (p CerebrasProvider) ProviderSuffix() string {
	return cerebrasProviderName
}

// Name returns the name of the Cerebras provider for debug purposes.
func (p CerebrasProvider) Name() string {
	return cerebrasProviderName
}
