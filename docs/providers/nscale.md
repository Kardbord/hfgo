# Nscale provider

`hfproviders.NscaleProvider` serves chat completion and text-to-image tasks
through the Hugging Face router. Like DeepInfra, Nscale mixes the two routing
mechanisms the provider layer offers: chat completions pin routing through the
model-string suffix (like the other chat partner providers), while text-to-image
resolves Nscale's own model identifiers through a model resolver (like WaveSpeed
and Fal AI).

This page covers only what is specific to Nscale; the provider layer's design
is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                        | Endpoint method         | Codec                       | Client method   |
| --------------------------- | ----------------------- | --------------------------- | --------------- |
| Chat completion (unary)     | `ChatEndpoint`          | `ChatCodec`                 | `Chat`          |
| Chat completion (streaming) | `ChatEndpoint`          | `ChatStreamCodec`           | `ChatStream`    |
| Text-to-image               | `TextToImageEndpoint`   | `TextToImageCodec`          | `GenerateImage` |

Structured message content — including the image parts vision-language models
take — passes through the shared chat codec unchanged, so VLM requests need no
extra wiring.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and chat needs no
[model resolver](../architecture.md#external-provider-model-resolution): the
router performs provider mapping server-side.

The task pipeline appends `:nscale` to the resolved model string unless the
model already carries an explicit suffix, so

```go
provider, err := hfproviders.NewNscaleProvider(hfproviders.NewStaticModelResolver(nil))
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("meta-llama/Llama-3.1-8B-Instruct"),
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

sends `"model": "meta-llama/Llama-3.1-8B-Instruct:nscale"` on the wire. Chat
never consults the resolver, so the static resolver in the example (which fails
every lookup) is fine for chat-only use; a `HubModelResolver` is only required
for the pipeline task below.

The `:nscale` suffix is a provider pin, not a preference: the router routes to
Nscale deterministically. Hugging Face's automatic failover — which applies to
unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred` policies —
does not reroute an explicit provider: if Nscale does not serve the model or
task, the router returns an error rather than running the request on another
provider. A request model that already carries a suffix is passed through
as-is, which opts out of the pin.

## Text-to-image

Text-to-image is served through Nscale's OpenAI-compatible images endpoint,
proxied by the router under the `nscale` path prefix
(`nscale/v1/images/generations`). Unlike chat, the request body carries
Nscale's own model identifier, so the provider resolves the Hugging Face model
ID you configure to Nscale's identifier through a
[model resolver](../architecture.md#external-provider-model-resolution) at
endpoint-construction time.

The codec encodes the canonical request into Nscale's flat JSON body: the
typed parameters are forwarded as-is — `Width`/`Height` are NOT folded into a
`size` field and `NumInferenceSteps` is NOT dropped, unlike the official
Python client — `ProviderParams` merge over the typed fields, and
`response_format`, `prompt`, and the mapped model are applied last so
`ProviderParams` entries cannot override them. Nscale-specific request
parameters such as `n` or `size` ride the request's `ProviderParams` field:

```go
image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene landscape with mountains at sunset",
	Parameters: &hftypes.TextToImageParameters{
		Width:             new(1024),
		Height:            new(1024),
		NumInferenceSteps: new(30),
	},
	ProviderParams: map[string]any{
		"n": 1,
	},
})
```

The response carries a base64-encoded image in `data[0].b64_json`; the codec
decodes it, reports the sniffed media type on `resp.MediaType`, and exposes
the response's `created` timestamp on `resp.ProviderMetadata` when present.
Because the image bytes arrive inside the JSON payload rather than as an HTTP
body, they carry no `Content-Type` header — the media type is inferred by
sniffing the bytes, so an unrecognized payload yields
`application/octet-stream`.

### Resolver options

The provider requires a model resolver to serve text-to-image. Two options
ship with the SDK:

- **`HubModelResolver`** — resolves mappings live against the Hub's
  `inferenceProviderMapping` data. Use it for general-purpose deployments;
  it is the default in the runnable examples.
- **`StaticModelResolver`** — pins explicit HF-to-Nscale mappings in code.
  Use it when you control the deployment and want no Hub lookup, or to map
  models the Hub does not yet list for Nscale.

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewNscaleProvider(resolver)
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("black-forest-labs/FLUX.1-schnell"),
	hfopts.WithProvider(provider),
)

image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene landscape with mountains at sunset",
})
```

A zero-value `NscaleProvider` is safe to construct — chat works without a
resolver — but its text-to-image endpoint methods return a configuration
`*hferrors.SDKError` instead of dereferencing a nil resolver. Pass
`hfproviders.NewStaticModelResolver(nil)` if you only need chat.

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own Nscale API key in
your Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
`:nscale` suffix stay unchanged — and your Nscale account is billed instead.
See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing its mixins

Like the other providers, Nscale is composed from per-task mixins:
`HuggingFaceChatMixin` — the same zero-byte mixin the default provider and the
chat-only partners use — plus `NscaleTextToImageMixin`, which carries an
exported `Resolver ModelResolver` field that maps the configured Hugging Face
model ID to Nscale's identifier at endpoint-construction time. Embed the
mixins for the tasks you serve and set the `Resolver` explicitly (or use
`NewNscaleProvider` to wire them all). The resolved mapping is attached to
the endpoint and the pipeline hands it to the codec; standalone codec use
falls back to the codec's own resolver. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the pattern.

## See also

- [examples/nscale](../../examples/nscale) — runnable programs for each
  supported task: chat in `basic`, `streaming`, and `conversation` variants,
  and text-to-image wired with both the Hub and static resolvers.
- [Nscale on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/nscale)
- [DeepInfra provider](./deepinfra.md) — the chat-routing convention the chat
  half follows.
- [Fal AI provider](./fal-ai.md) — the resolver-backed pipeline-task pattern
  the other half follows.
- [architecture.md](../architecture.md)
