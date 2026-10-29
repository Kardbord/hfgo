# OVHcloud provider

`hfproviders.OVHcloudProvider` pins chat completions to OVHcloud AI Endpoints
through the Hugging Face router. OVHcloud serves OpenAI-compatible chat
completion for both conversational LLMs and vision-language models, so the
provider reuses the default Hugging Face chat endpoint and codecs and changes
only the routing: the router receives the model ID with the `:ovhcloud`
suffix appended (e.g. `openai/gpt-oss-120b:ovhcloud`).

This page covers only what is specific to OVHcloud; the provider layer's
design is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                        | Endpoint method | Codec             | Client method |
| --------------------------- | --------------- | ----------------- | ------------- |
| Chat completion (unary)     | `ChatEndpoint`  | `ChatCodec`       | `Chat`        |
| Chat completion (streaming) | `ChatEndpoint`  | `ChatStreamCodec` | `ChatStream`  |

Chat is the only task OVHcloud serves: casting the provider to any
pipeline-task interface (e.g. `SummarizationProvider`) fails with a
configuration `*hferrors.SDKError`, exactly as with the default provider.
Structured message content — including the image parts vision-language models
take — passes through the shared chat codec unchanged, so VLM requests need no
extra wiring.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and OVHcloud needs no
[model resolver](../architecture.md#external-provider-model-resolution) — the
router performs provider mapping server-side, unlike
[WaveSpeed](./wavespeed.md), whose pipeline endpoints need the HF-to-provider
identifier at construction time.

The task pipeline appends `:ovhcloud` to the resolved model string unless the
model already carries an explicit suffix, so

```go
client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("openai/gpt-oss-120b"),
	hfopts.WithProvider(hfproviders.NewOVHcloudProvider()),
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

sends `"model": "openai/gpt-oss-120b:ovhcloud"` on the wire.

The `:ovhcloud` suffix is a provider pin, not a preference: the router routes
to OVHcloud deterministically. Hugging Face's automatic failover — which
applies to unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred`
policies — does not reroute an explicit provider: if OVHcloud does not serve
the model or task, the router returns an error such as
`400 Model not supported by provider ovhcloud` rather than running the request
on another provider.

A request model that already carries a suffix is passed through as-is, which
opts out of the pin. Pass e.g. `openai/gpt-oss-120b:preferred` as the
request model to let the router select (and fail over between) providers
instead. See the
[Inference Providers docs](https://huggingface.co/docs/inference-providers/main/en/index)
for suffix selection behavior.

## Reusing its mixins

OVHcloud adds no mixins of its own: it embeds
[`HuggingFaceChatMixin`](./huggingface.md#reusing-its-mixins) — the same
zero-byte mixin the default provider uses — and supplies only its identity
(`Provider.Name`, `ProviderSuffix`). To build a custom provider that serves
only chat, embed `HuggingFaceChatMixin` and implement the two identity methods
the same way; see
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for the
pattern.

## See also

- [examples/ovhcloud](../../examples/ovhcloud) — runnable chat programs in
  `basic`, `streaming`, and `conversation` variants.
- [Hugging Face provider](./huggingface.md) — the default provider whose chat
  mixin and codecs OVHcloud reuses.
- [OVHcloud AI Endpoints on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/ovhcloud)
- [architecture.md](../architecture.md)
