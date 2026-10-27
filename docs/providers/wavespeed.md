# WaveSpeed provider

`hfproviders.WavespeedProvider` routes text-to-video, text-to-image, and
image-to-image tasks through WaveSpeed's inference platform via the Hugging
Face router. The Hugging Face provider does not serve these tasks, so requests
need an explicit provider.

This page covers only what is specific to WaveSpeed; the provider layer's
design is documented in
[architecture.md](../architecture.md#provider-abstraction).

## Model resolution

Callers configure the Hugging Face model ID they know; the provider translates
it to WaveSpeed's own model identifier at endpoint-construction time via a
[model resolver](../architecture.md#external-provider-model-resolution). The
constructor refuses a nil resolver, and the zero-value provider fails endpoint
resolution with a configuration `*hferrors.SDKError` rather than panicking.

### Hub resolution (recommended)

`hfproviders.HubModelResolver` queries the Hub's `inferenceProviderMapping`
data for the configured model and caches results per resolver instance, so no
static mapping table is required. See
[architecture.md](../architecture.md#external-provider-model-resolution) for
the resolver options (`WithHubToken`, `WithHubCacheTTL`, `WithHubMaxEntries`,
`WithHubStaleOnError`, and the rest) and the caching semantics.

### Static resolution

For deployments that pin the HF-to-provider mapping themselves or run without
network access to the Hub, use `hfproviders.StaticModelResolver`:

```go
mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]string{
	{
		Provider: "wavespeed",
		Task:     hfproviders.TaskTextToImage,
		Model:    "black-forest-labs/FLUX.1-dev",
	}: "wavespeed-ai/flux-dev",
})

provider, err := hfproviders.NewWavespeedProvider(mapper)
if err != nil {
	log.Fatal(err)
}
```

## Supported tasks

| Task           | Endpoint method        | Codec                    | Client method        |
| -------------- | ---------------------- | ------------------------ | -------------------- |
| Text to video  | `TextToVideoEndpoint`  | `TextToVideoCodec`       | `GenerateVideo`      |
| Text to image  | `TextToImageEndpoint`  | `TextToImageCodec`       | `GenerateImage`      |
| Image to image | `ImageToImageEndpoint` | `ImageToImageCodec`      | `GenerateImageToImage` |

Each task returns raw binary bytes together with the media type the provider
advertised (`response.MediaType`). Requests are encoded as JSON; responses are
read from the raw body. Generated videos and images can exceed the SDK's
default 1 MiB response cap, so raise it with
`hfopts.WithMaxResponseBodyBytes` when needed.

Quick start with the Hub resolver:

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewWavespeedProvider(resolver)
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("black-forest-labs/FLUX.1-dev"),
	hfopts.WithProvider(provider),
	hfopts.WithMaxResponseBodyBytes(16<<20),
)

image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene mountain landscape at sunset",
})
```

## Reusing its mixins

Like the Hugging Face provider, WaveSpeed is composed from per-task mixins:
`WavespeedTextToVideoMixin`, `WavespeedTextToImageMixin`, and
`WavespeedImageToImageMixin`. Unlike the Hugging Face mixins, each carries an
exported `Resolver ModelResolver` field that maps the configured Hugging Face
model ID to WaveSpeed's identifier at endpoint-construction time — embed the
mixins for the tasks you serve and set each `Resolver` explicitly (or use
`NewWavespeedProvider` to wire all three). The text-to-image mixin reuses the
shared Hugging Face text-to-image codec. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for the
pattern.

## See also

- [examples/wavespeed](../../examples/wavespeed) — runnable programs for each
  supported task.
- [WaveSpeed on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/wavespeed)
- [architecture.md](../architecture.md)
