package hfproviders

// groqProviderName is the Groq provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":groq" provider pin.
const groqProviderName = "groq"

// GroqProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":groq" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Groq needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":groq" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Groq does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type GroqProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that GroqProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*GroqProvider)(nil)
	_ ChatStreamProvider = (*GroqProvider)(nil)
)

// NewGroqProvider returns a GroqProvider that serves
// chat completion tasks.
func NewGroqProvider() GroqProvider {
	return GroqProvider{}
}

// ProviderSuffix returns the "groq" provider suffix for
// appending to the model selection in chat completion tasks.
func (p GroqProvider) ProviderSuffix() string {
	return groqProviderName
}

// Name returns the name of the Groq provider for debug purposes.
func (p GroqProvider) Name() string {
	return groqProviderName
}
