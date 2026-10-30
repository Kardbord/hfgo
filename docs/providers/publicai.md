# Public AI provider

`hfproviders.PublicAIProvider` pins chat completions to Public AI through the
Hugging Face router. Public AI serves OpenAI-compatible chat completion for
conversational LLMs, so the provider reuses the default Hugging Face chat
endpoint and codecs and changes only the routing: the router receives the
model ID with the `:publicai` suffix appended (e.g.
`speakleash/Bielik-11B-v3.0-Instruct:publicai`).

This page covers only what is specific to Public AI; the provider layer's
design is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                        | Endpoint method | Codec             | Client method |
| --------------------------- | --------------- | ----------------- | ------------- |
| Chat completion (unary)     | `ChatEndpoint`  | `ChatCodec`       | `Chat`        |
| Chat completion (streaming) | `ChatEndpoint`  | `ChatStreamCodec` | `ChatStream`  |

Chat is the only task Public AI serves: casting the provider to any
pipeline-task interface (e.g. `SummarizationProvider`) fails with a
configuration `*hferrors.SDKError`, exactly as with the default provider.
Structured message content passes through the shared chat codec unchanged.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and Public AI needs no
[model resolver](../architecture.md#external-provider-model-resolution) — the
router performs provider mapping server-side, unlike
[WaveSpeed](./wavespeed.md), whose pipeline endpoints need the HF-to-provider
identifier at construction time.

The task pipeline appends `:publicai` to the resolved model string unless the
model already carries an explicit suffix, so

```go
client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("speakleash/Bielik-11B-v3.0-Instruct"),
	hfopts.WithProvider(hfproviders.NewPublicAIProvider()),
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

sends `"model": "speakleash/Bielik-11B-v3.0-Instruct:publicai"` on the wire.

The `:publicai` suffix is a provider pin, not a preference: the router routes
to Public AI deterministically. Hugging Face's automatic failover — which
applies to unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred`
policies — does not reroute an explicit provider: if Public AI does not serve
the model or task, the router returns an error such as
`400 Model not supported by provider publicai` rather than running the request
on another provider.

A request model that already carries a suffix is passed through as-is, which
opts out of the pin. Pass e.g. `speakleash/Bielik-11B-v3.0-Instruct:preferred`
as the request model to let the router select (and fail over between)
providers instead. See the
[Inference Providers docs](https://huggingface.co/docs/inference-providers/main/en/index)
for suffix selection behavior.

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own Public AI API key in
your Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
`:publicai` suffix stay unchanged — and your Public AI account is billed
instead. See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing its mixins

Public AI adds no mixins of its own: it embeds
[`HuggingFaceChatMixin`](./huggingface.md#reusing-its-mixins) — the same
zero-byte mixin the default provider uses — and supplies only its identity
(`Provider.Name`, `ProviderSuffix`). To build a custom provider that serves
only chat, embed `HuggingFaceChatMixin` and implement the two identity methods
the same way; see
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for the
pattern.

## See also

- [examples/publicai](../../examples/publicai) — runnable chat programs in
  `basic`, `streaming`, and `conversation` variants.
- [Hugging Face provider](./huggingface.md) — the default provider whose chat
  mixin and codecs Public AI reuses.
- [PublicAI on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/publicai)
- [architecture.md](../architecture.md)
