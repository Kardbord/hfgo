package hfproviders

// novitaProviderName is the Novita provider slug and the model-suffix token
// appended to chat model IDs, matching the router's ":novita" provider pin.
const novitaProviderName = "novita"

// NovitaProvider serves chat completion tasks through the Hugging Face router
// by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":novita" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], Novita needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":novita" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Novita does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// Structured message content passes through the shared chat codec unchanged,
// so vision-language requests to Novita's multimodal models need no extra
// wiring, matching the official Hugging Face clients and provider docs.
//
// Novita also serves text-to-video and text-generation on the official
// clients' direct-API routes, but neither is exposed here: at the time of
// writing, no Novita text-to-video model appears in the Hub's provider
// mapping, and this SDK has no text-generation task. If Novita text-to-video
// support lands in the mapping, this provider's docs and godoc will need
// updating.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type NovitaProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that NovitaProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*NovitaProvider)(nil)
	_ ChatStreamProvider = (*NovitaProvider)(nil)
)

// NewNovitaProvider returns a NovitaProvider that serves
// chat completion tasks.
func NewNovitaProvider() NovitaProvider {
	return NovitaProvider{}
}

// ProviderSuffix returns the "novita" provider suffix for
// appending to the model selection in chat completion tasks.
func (p NovitaProvider) ProviderSuffix() string {
	return novitaProviderName
}

// Name returns the name of the Novita provider for debug purposes.
func (p NovitaProvider) Name() string {
	return novitaProviderName
}
