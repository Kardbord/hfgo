# Replicate provider

`hfproviders.ReplicateProvider` routes text-to-image, image-to-image,
text-to-video, and speech-recognition tasks through Replicate's inference
platform via the Hugging Face router. The Hugging Face provider does not serve
these tasks, so requests need an explicit provider.

This page covers only what is specific to Replicate; the provider layer's
design is documented in
[architecture.md](../architecture.md#provider-abstraction).

## Model resolution

Callers configure the Hugging Face model ID they know; the provider translates
it to Replicate's own model identifier at endpoint-construction time via a
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
mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]hfproviders.ModelMapping{
	{
		Provider: "replicate",
		Task:     hfproviders.TaskTextToImage,
		Model:    "black-forest-labs/FLUX.1-dev",
	}: {ProviderModelID: "black-forest-labs/flux-dev"},
})

provider, err := hfproviders.NewReplicateProvider(mapper)
if err != nil {
	log.Fatal(err)
}
```

## Supported tasks

| Task                     | Endpoint method                      | Codec                          | Client method           |
| ------------------------ | ------------------------------------ | ------------------------------ | ----------------------- |
| Text to image            | `TextToImageEndpoint`                | `TextToImageCodec`             | `GenerateImage`         |
| Image to image           | `ImageToImageEndpoint`               | `ImageToImageCodec`            | `GenerateImageToImage`  |
| Text to video            | `TextToVideoEndpoint`                | `TextToVideoCodec`             | `GenerateVideo`         |
| Speech recognition       | `SpeechRecognitionEndpoint`          | `SpeechRecognitionCodec`       | `RecognizeSpeech`       |

Each task returns the generated media together with the media type the provider
advertised (`response.MediaType`), falling back to the type byte-sniffed from
the payload when the advertised type is missing, garbled, or generic (CDNs
commonly label every blob `application/octet-stream`). Replicate does not serve
these tasks with
the Hugging Face pipeline wire format:

- **Requests** are encoded as Replicate's flat prediction payload. The prompt
  and every generation parameter live at the top level — the canonical
  request's `Input` becomes `prompt` (text tasks) or `image` (image-to-image,
  wrapped as a `data:` URI when it is raw base64; matching the official
  Hugging Face clients), and the typed parameters are merged in. Replicate
  routes via the model's `version` field: the resolved mapping's versioned
  route is used so model updates are pinned to the Hub mapping.
- **Responses** follow Replicate's asynchronous prediction API. The SDK
  submits the prediction with `Prefer: wait` to hold the request open while
  the prediction starts, polls past that window while it runs (2 s initial
  wait, growing by 1 s per poll up to 5 s; transient poll failures are
  retried, a failed prediction surfaces the provider's error), and downloads
  the first output. The exchange is transparent to callers and is bounded by
  the request context, so give long-running generations a generous deadline
  (e.g. `hfopts.WithContext`).

Generated videos and images can exceed the SDK's default 1 MiB response cap,
so raise it with `hfopts.WithMaxResponseBodyBytes` when needed.

### Provider-specific request parameters

Replicate models take provider-native parameters, not the canonical Hugging
Face ones: e.g. FLUX models accept `aspect_ratio`, `output_format`, and
`output_quality`, and reject unknown fields — such as the canonical
`num_frames` — with a 400. The canonical request types do not cover every
knob Replicate models expose, so every request DTO carries an advanced escape
hatch for this:

```go
image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene mountain landscape at sunset",
	ProviderParams: map[string]any{
		"output_format": "png",
	},
})
```

`ProviderParams` entries are merged into the flat body after the typed
parameters, so entries override typed fields — including the codec-provided
`image` entry (for image-to-image models) and the injected `loras` entry
(last write wins). This mirrors the `extra_body` argument of the official
Python client.

### Provider-specific response metadata

Responses may carry provider-specific details in `ProviderMetadata`
(e.g. `id`, `model`, `created_at`, `metrics.predict_time`). Providers populate
the map only when there is something to add; it is `nil` otherwise.

Quick start with the Hub resolver:

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewReplicateProvider(resolver)
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

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own Replicate API key in
your Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
resolved endpoint stay unchanged — and your Replicate account is billed
instead. See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing its mixins

Like the Hugging Face provider, Replicate is composed from per-task mixins:
`ReplicateTextToImageMixin`, `ReplicateImageToImageMixin`,
`ReplicateTextToVideoMixin`, and
`ReplicateSpeechRecognitionMixin`. Unlike the Hugging Face mixins,
each carries an exported `Resolver ModelResolver` field that maps the
configured Hugging Face model ID to Replicate's identifier at
endpoint-construction time — embed the mixins for the tasks you serve and set
each `Resolver` explicitly (or use `NewReplicateProvider` to wire all four).
The resolved mapping is attached to the endpoint and the pipeline hands it to
the codec, so adapter-model payloads get their LoRA injection without a second
lookup; standalone codec use falls back to the codec's own resolver. Each
mixin serves an `AsyncCodec`: its codecs encode Replicate's flat body and
complete the wait-poll-download exchange themselves. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for the
pattern.

## See also

- [examples/replicate](../../examples/replicate) — runnable programs for each
  supported task, in `hub-resolver` variants.
- [DeepInfra provider](./deepinfra.md) — the Hub-resolver quick start this
  guide follows for pinning an explicit provider.
- [Replicate on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/replicate)
- [architecture.md](../architecture.md)
