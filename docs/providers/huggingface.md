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
unless the model string carries an explicit provider suffix such as
`model:sambanova`, `model:fastest`, `model:cheapest`, or `model:preferred`;
see the [Inference Providers docs](https://huggingface.co/docs/inference-providers/main/en/index)
for selection behavior. Because chat payloads carry the model themselves, chat
does not use a [model resolver](../architecture.md#external-provider-model-resolution)
and the endpoint is model-independent.

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
- [Package reference](https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders)
- [Hugging Face Inference Providers documentation](https://huggingface.co/docs/inference-providers/main/en/index)
- [architecture.md](../architecture.md)
