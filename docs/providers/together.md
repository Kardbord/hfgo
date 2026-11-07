# Together provider

`hfproviders.TogetherProvider` serves chat completions (unary and streaming,
including vision-language requests) through the Hugging Face router, pinned
to Together with the `:together` model suffix like the other chat partner
providers. It reuses the shared chat mixin — endpoint and codecs — because
Together's chat-completions API consumes the canonical OpenAI-compatible
wire format the mixin emits.

This page covers only what is specific to Together; the provider layer's
design is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                        | Endpoint method | Codec             | Client method |
| --------------------------- | --------------- | ----------------- | ------------- |
| Chat completion (unary)     | `ChatEndpoint`  | `ChatCodec`       | `Chat`        |
| Chat completion (streaming) | `ChatEndpoint`  | `ChatStreamCodec` | `ChatStream`  |

Structured message content — including the image parts vision-language
models take — passes through the chat codec unchanged, so VLM requests need
no extra wiring. Casting the provider to any task interface it does not
serve fails with a configuration `*hferrors.SDKError`, exactly as with the
default provider.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and Together needs **no model
resolver**: the router performs provider mapping server-side.

The task pipeline appends `:together` to the resolved model string unless the
model already carries an explicit suffix, so

```go
provider := hfproviders.NewTogetherProvider()

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("deepseek-ai/DeepSeek-V4.1-Flash"),
	hfopts.WithProvider(provider),
)

response, err := client.Chat(context.Background(), hftypes.ChatRequest{
	Messages: []hftypes.ChatMessage{{
		Role: "user",
		Content: hftypes.ChatMessageContent{
			Text: new("What is the capital of France?"),
		},
	}},
})
```

sends `"model": "deepseek-ai/DeepSeek-V4.1-Flash:together"` on the wire.

The `:together` suffix is a provider pin, not a preference: the router routes
to Together deterministically. Hugging Face's automatic failover — which
applies to unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred`
policies — does not reroute an explicit provider: if Together does not serve
the model or task, the router returns an error rather than running the request
on another provider. A request model that already carries a suffix is passed
through as-is, which opts out of the pin.

## Structured outputs (json_schema response formats)

Structured output for Together uses OpenAI's nested `response_format`
envelope — `{"type": "json_schema", "json_schema": {"schema": ...}}` —
which is exactly what `hftypes.ChatRequest` marshals to: `name`,
`description`, `schema`, and `strict` all travel to Together as the caller
set them. (`text`, `json_object`, and provider-specific format types pass
through unchanged too.)

HF's official JS and Python clients send a flattened
`{"type": "json_schema", "schema": ...}` shape instead; Together's current
API rejects it with `missing field \`json_schema\``. Callers who need that
shape (or any other provider-specific shape) can pass it via
`hftypes.ChatRequest.ProviderParams`, which overrides the typed
`response_format` on the wire.

> **Known upstream breakage:** structured output over the Hugging Face
> router currently fails regardless of the client. The router's Together
> mapping flattens `response_format` before forwarding, and Together rejects
> the flattened shape — a routed `json_schema` request returns 400
> `json_parse_error` ("missing field `json_schema`"), while the same request
> direct to `api.together.xyz` with the nested envelope succeeds. The live
> integration test `TestChatCompletion_TogetherJSONSchemaLiveAPI` is skipped
> until the router forwards the nested envelope untouched; the SDK needs no
> change when that fix ships.

```go
response, err := client.Chat(context.Background(), hftypes.ChatRequest{
	Messages: []hftypes.ChatMessage{{
		Role: "user",
		Content: hftypes.ChatMessageContent{
			Text: new("What is the capital of France? Answer as a JSON object."),
		},
	}},
	ResponseFormat: &hftypes.ChatResponseFormat{
		Type: hftypes.ResponseFormatTypeJSONSchema,
		JSONSchema: &hftypes.ChatJSONSchemaConfig{
			Name:   "capital",
			Strict: new(true),
			Schema: json.RawMessage(
				`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"]}`,
			),
		},
	},
})
```

sends `"response_format": {"type":"json_schema","json_schema":{"name":"capital","strict":true,"schema":{...}}}` on the wire.
Callers always build and receive the canonical `hftypes` shapes.

Responses decode with the default provider's schema; Together omits
`system_fingerprint` from some responses and the canonical `hftypes` treat it
as optional, so no adaptation is needed.

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own Together API key in
your Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
`:together` suffix stay unchanged — and your Together account is billed
instead. See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing the chat mixin

`TogetherProvider` embeds `HuggingFaceChatMixin` — the same zero-byte mixin
the default provider and the chat-only partners use — and inherits its
model-independent chat endpoint and codecs, including the `ProviderParams`
merge: entries ride the body's top level with the resolved `"model"`
protected. A custom provider embedding `HuggingFaceChatMixin` gets the same
wire behavior. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the general pattern.

## See also

- [examples/together](../../examples/together) — runnable programs: chat in
  `basic`, `streaming`, and `conversation` variants.
- [Together on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/together)
- [Groq provider](./groq.md) — the chat-routing convention this provider
  follows.
- [architecture.md](../architecture.md)
