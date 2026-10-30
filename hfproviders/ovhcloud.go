package hfproviders

// ovhcloudProviderName is the OVHcloud provider slug and the model-suffix
// token appended to chat model IDs, matching the router's ":ovhcloud"
// provider pin.
const ovhcloudProviderName = "ovhcloud"

// OVHcloudProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin], as the chat task has the same OpenAI-compatible
// request/response schema across providers: only the routing differs,
// expressed by appending the ":ovhcloud" provider suffix to the model ID.
//
// Unlike [WavespeedProvider], OVHcloud needs no [ModelResolver]: the chat
// endpoint is model-independent, and the router performs provider mapping
// server-side. The ":ovhcloud" suffix is a deterministic provider pin — the
// router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model OVHcloud does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin.
//
// The zero value is ready to use. The provider is immutable and safe for
// concurrent use.
type OVHcloudProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that OVHcloudProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*OVHcloudProvider)(nil)
	_ ChatStreamProvider = (*OVHcloudProvider)(nil)
)

// NewOVHcloudProvider returns an OVHcloudProvider that serves
// chat completion tasks.
func NewOVHcloudProvider() OVHcloudProvider {
	return OVHcloudProvider{}
}

// ProviderSuffix returns the "ovhcloud" provider suffix for
// appending to the model selection in chat completion tasks.
func (p OVHcloudProvider) ProviderSuffix() string {
	return ovhcloudProviderName
}

// Name returns the name of the OVHcloud provider for debug purposes.
func (p OVHcloudProvider) Name() string {
	return ovhcloudProviderName
}
