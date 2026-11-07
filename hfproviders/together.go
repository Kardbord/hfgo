package hfproviders

// togetherProviderName is the Together provider slug and the model-suffix
// token appended to chat model IDs, matching the router's provider
// identifiers.
const togetherProviderName = "together"

// TogetherProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin]'s endpoint and codecs because the OpenAI-compatible
// schema travels in the request body: the router performs provider mapping
// server-side, so no [ModelResolver] is required. Routing is pinned to
// Together by appending the ":together" provider suffix to the model ID.
// The suffix is a deterministic pin — the router's automatic failover applies
// to unsuffixed models and routing policies (":fastest", ":cheapest",
// ":preferred"), not to an explicit provider — so a model Together does not
// serve fails rather than running on another provider. Pass a request model
// with its own suffix (e.g. "model:preferred") to opt out of the pin.
//
// Structured output rides OpenAI's nested json_schema response_format
// envelope — {"type":"json_schema","json_schema":{"schema":...}} — which is
// exactly what [hftypes.ChatRequest] marshals to, so requests are sent as
// built: name, description, schema, and strict all travel to Together as
// the caller set them. (HF's official JS and Python clients instead send a
// flattened {"type":"json_schema","schema":...} shape, which Together's
// current API rejects with "missing field json_schema".) Callers who need
// that shape, or any other provider-specific response_format, can pass it
// via [hftypes.ChatRequest.ProviderParams], which overrides the typed field
// on the wire. Responses decode like the default provider's, including
// vision-language content.
//
// The zero value is ready to use, as with the other chat partner providers.
type TogetherProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that TogetherProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*TogetherProvider)(nil)
	_ ChatStreamProvider = (*TogetherProvider)(nil)
)

// NewTogetherProvider returns a TogetherProvider that serves chat completion
// (unary and streaming) pinned to Together through the ":together" model
// suffix.
func NewTogetherProvider() TogetherProvider {
	return TogetherProvider{}
}

// Name returns the name of the Together provider for debug purposes.
func (p TogetherProvider) Name() string {
	return togetherProviderName
}

// ProviderSuffix returns the "together" provider suffix for appending to the
// model selection in chat completion tasks.
func (p TogetherProvider) ProviderSuffix() string {
	return togetherProviderName
}
