# Hugging Face provider

`hfproviders.HuggingFaceProvider` is the SDK's default inference provider. It
speaks the Hugging Face Inference API wire format directly, so most users never
configure it explicitly. This page covers only what is specific to it; the
[provider layer's design](../architecture.md#per-task-provider-mixins) is
documented in architecture.md.

## Supported tasks

The provider serves every task the SDK models except text-to-video and
image-to-image, which the Hugging Face router does not expose through the
canonical task endpoints. Those tasks require a partner provider such as
[WaveSpeed](./wavespeed.md).

Pipeline tasks (fill-mask, text-to-image, and the rest) use the router's task
endpoints and the canonical JSON wire format, with the model in the endpoint
path.

## Chat routing

Chat completions are routed through the Hugging Face router's
OpenAI-compatible endpoint. The router selects a partner provider server-side
based on the model string's suffix (see the
[Inference Providers docs](https://huggingface.co/docs/inference-providers/main/en/index)):

- **No suffix, or `:fastest`** — the fastest available provider; if it becomes
  unavailable, the router automatically fails over to another provider.
- **`:cheapest` / `:preferred`** — a routing policy (lowest output-token
  price / your Inference Provider settings order); the router still fails over
  to another provider when needed.
- **A provider name (e.g. `:baseten`)** — that provider, deterministically.
  There is no cross-provider fallback: if the pinned provider does not serve
  the model or task, the router returns an error such as
  `400 Model not supported by provider <name>`.

Because chat payloads carry the model themselves, chat does not use a
[model resolver](../architecture.md#external-provider-model-resolution) and
the endpoint is model-independent. Built-in providers automate one such pin:
each appends its `:<provider>` suffix to the model string; the
[docs index](../README.md) lists the chat-pinned provider guides.

## Billing

Hugging Face Inference is billed directly on your Hugging Face account:
monthly included credits apply automatically, and usage beyond them is
pay-as-you-go. Partner providers differ — routed requests are billed on your
HF account only for pay-as-you-go compatible providers, and a custom provider
key in your account settings bills the provider directly instead. See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) and
the billing note in each partner-provider guide.

## Reusing its mixins

`hfproviders` exposes one zero-byte mixin per task
(`HuggingFaceTextClassificationMixin`, `HuggingFaceTextToImageMixin`, and so
on) that pairs the built-in endpoint and codec for that task. Embed the mixins
for the tasks your custom provider serves and supply your own
`Provider.Name` and `Provider.ProviderSuffix`. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for the
pattern and its trade-offs.

## See also

- [WaveSpeed provider](./wavespeed.md) — the built-in partner-provider example.
- [Fal AI provider](./fal-ai.md) — media and speech generation tasks through
  fal's platform.
- [Baseten provider](./baseten.md) — chat completions pinned to Baseten through
  the router's `:baseten` suffix.
- [Cerebras provider](./cerebras.md) — chat completions pinned to Cerebras
  through the router's `:cerebras` suffix.
- [Cohere provider](./cohere.md) — chat completions pinned to Cohere through
  the router's `:cohere` suffix.
- [Featherless AI provider](./featherless-ai.md) — chat completions pinned to
  Featherless AI through the router's `:featherless-ai` suffix.
- [Fireworks provider](./fireworks-ai.md) — chat completions pinned to
  Fireworks through the router's `:fireworks-ai` suffix.
- [Groq provider](./groq.md) — chat completions pinned to Groq through the
  router's `:groq` suffix.
- [OVHcloud provider](./ovhcloud.md) — chat completions pinned to OVHcloud AI
  Endpoints through the router's `:ovhcloud` suffix.
- [Public AI provider](./publicai.md) — chat completions pinned to Public AI
  through the router's `:publicai` suffix.
- [Z.ai provider](./zai-org.md) — chat completions pinned to Z.ai through the
  router's `:zai-org` suffix.
- [Package reference](https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders)
- [Hugging Face Inference Providers documentation](https://huggingface.co/docs/inference-providers/main/en/index)
- [architecture.md](../architecture.md)
