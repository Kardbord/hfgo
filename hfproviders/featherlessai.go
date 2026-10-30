package hfproviders

// featherlessAIProviderName is the Featherless AI provider slug and the
// model-suffix token appended to chat model IDs, matching the router's
// ":featherless-ai" provider pin.
const featherlessAIProviderName = "featherless-ai"

// FeatherlessAIProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":featherless-ai" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Featherless AI needs no [ModelResolver]: the
// chat endpoint is model-independent, and the router performs provider mapping
// server-side. The ":featherless-ai" suffix is a deterministic provider pin —
// the router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Featherless AI does not serve fails rather than
// running on another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type FeatherlessAIProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that FeatherlessAIProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*FeatherlessAIProvider)(nil)
	_ ChatStreamProvider = (*FeatherlessAIProvider)(nil)
)

// NewFeatherlessAIProvider returns a FeatherlessAIProvider that serves
// chat completion tasks.
func NewFeatherlessAIProvider() FeatherlessAIProvider {
	return FeatherlessAIProvider{}
}

// ProviderSuffix returns the "featherless-ai" provider suffix for
// appending to the model selection in chat completion tasks.
func (p FeatherlessAIProvider) ProviderSuffix() string {
	return featherlessAIProviderName
}

// Name returns the name of the Featherless AI provider for debug purposes.
func (p FeatherlessAIProvider) Name() string {
	return featherlessAIProviderName
}
