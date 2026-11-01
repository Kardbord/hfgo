# Together provider

`hfproviders.TogetherProvider` serves chat completions (unary and streaming,
including vision-language requests) through the Hugging Face router, pinned
to Together with the `:together` model suffix like the other chat partner
providers. It differs from them in one respect: its chat requests go through
a provider-owned codec that adapts structured-output configuration to
Together's wire shape.

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

response, err := client.Chat(hftypes.ChatRequest{
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

Structured output for Together rides a flattened `response_format` shape —
`{"type": "json_schema", "schema": ...}` — rather than OpenAI's nested
`{"type": "json_schema", "json_schema": {"schema": ...}}` envelope: both
official Hugging Face clients send the flattened shape — even when routed
through Hugging Face — and the router forwards request bodies untouched, so
the rewrite is client-side work. This provider's chat codecs perform it
exactly as the official clients do: a `json_schema` response format is
unwrapped onto the wire and its `name` and `strict` fields are dropped. A
`json_schema` format whose envelope carries no `schema` is passed through
unchanged (matching the official clients), as are `text`, `json_object`, and
provider-specific format types.

```go
response, err := client.Chat(hftypes.ChatRequest{
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

sends `"response_format": {"type":"json_schema","schema":{...}}` on the wire.
Callers always build and receive the canonical `hftypes` shapes; the rewrite
is a wire-format detail that never surfaces.

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
the default provider and the chat-only partners use — for the
model-independent chat endpoint, and shadows the mixin's `ChatCodec` and
`ChatStreamCodec` accessors with the Together-specific codecs described
above. A custom provider embedding only `HuggingFaceChatMixin` gets the
default envelope shape — the shape the official clients rewrite away, which
is why Together-bound structured output needs the special codec. The Together
codecs themselves are unexported: external providers needing the flattened
rewrite should embed `TogetherProvider` (overriding only the identity
methods) to inherit its codec accessors, or re-implement the rewrite over
`JSONCodec`. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the general pattern.

## See also

- [examples/together](../../examples/together) — runnable programs: chat in
  `basic`, `streaming`, and `conversation` variants.
- [Together on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/together)
- [Groq provider](./groq.md) — the chat-routing convention this provider
  follows, without the codec twist.
- [architecture.md](../architecture.md)
