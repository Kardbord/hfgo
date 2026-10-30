# Fal AI provider

`hfproviders.FalAIProvider` routes text-to-image, image-to-image,
text-to-video, image-segmentation, and speech-recognition tasks through
fal's generative media inference platform via the Hugging Face router. The
Hugging Face provider does not serve these tasks, so requests need an
explicit provider.

This page covers only what is specific to Fal AI; the provider layer's
design is documented in
[architecture.md](../architecture.md#provider-abstraction).

## Model resolution

Callers configure the Hugging Face model ID they know; the provider
translates it to fal's own model identifier at endpoint-construction time via
a [model resolver](../architecture.md#external-provider-model-resolution). The
constructor refuses a nil resolver, and the zero-value provider fails endpoint
resolution with a configuration `*hferrors.SDKError` rather than panicking.

### Hub resolution (recommended)

`hfproviders.HubModelResolver` queries the Hub's `inferenceProviderMapping`
data for the configured model and caches results per resolver instance, so no
static mapping table is required. See
[architecture.md](../architecture.md#external-provider-model-resolution) for
the resolver options (`WithHubToken`, `WithHubCacheTTL`, `WithHubMaxEntries`,
and `WithHubStaleOnError`) and the caching semantics.

### Static resolution

For deployments that pin the HF-to-provider mapping themselves or run without
network access to the Hub, use `hfproviders.StaticModelResolver`:

```go
mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]hfproviders.ModelMapping{
	{
		Provider: "fal-ai",
		Task:     hfproviders.TaskTextToImage,
		Model:    "krea/Krea-2-Turbo",
	}: {ProviderModelID: "fal-ai/krea-2/turbo"},
})

provider, err := hfproviders.NewFalAIProvider(mapper)
if err != nil {
	log.Fatal(err)
}
```

## Supported tasks

| Task                | Endpoint method                | Codec                    | Client method        |
| ------------------- | ------------------------------ | ------------------------ | -------------------- |
| Text to image       | `TextToImageEndpoint`          | `TextToImageCodec`       | `GenerateImage`      |
| Image to image      | `ImageToImageEndpoint`         | `ImageToImageCodec`      | `GenerateImageToImage` |
| Text to video       | `TextToVideoEndpoint`          | `TextToVideoCodec`       | `GenerateVideo`      |
| Image segmentation  | `ImageSegmentationEndpoint`    | `ImageSegmentationCodec` | `SegmentImage`       |
| Speech recognition  | `SpeechRecognitionEndpoint`    | `SpeechRecognitionCodec` | `RecognizeSpeech`    |

Media tasks return the generated media together with the media type the
provider advertised (`response.MediaType`). fal does not serve these tasks
with the Hugging Face pipeline wire format:

- **Requests** are encoded as fal's flat JSON body. The prompt and every
  generation parameter live at the top level — the canonical request's
  `Input` becomes `prompt` (text tasks), `image_url` **and** `image_urls`
  (image-to-image, the dual shape mirrors the official Hugging Face clients
  and satisfies endpoints that require one or the other), `image_url`
  (image-segmentation), or `audio_url` (speech recognition). Raw base64
  inputs are wrapped as `data:` URIs; data URIs and HTTP(S) URLs pass through
  unchanged. Canonical `width`/`height` parameters fold into fal's
  `image_size` object and `target_size` renames to it, exactly as the
  official Python client prepares payloads.
- **Responses** follow fal's queue API for every task. The SDK submits the
  task, polls the task's status URL until it reaches the terminal
  `COMPLETED` status (500 ms initial wait growing toward 5 s; transient poll
  failures are retried and a server `Retry-After` is honored), fetches the
  result, and — for media tasks — downloads the first output. A task that
  fails reports `COMPLETED` with an error payload; the SDK surfaces that as
  an `*hferrors.SDKError` of kind `provider`. The exchange is transparent to
  callers and is bounded by the request context, so give long-running
  generations a generous deadline (e.g. `hfopts.WithContext`).

> **Note:** Both official Hugging Face clients call fal's *synchronous*
> endpoint for speech recognition; this SDK routes every fal task through
> the queue for one uniform exchange, which additionally inherits fal's
> transparent retry of transient runner failures. If a speech model ever
> rejects the queue route, that task is the one to revisit.

### Adapter (LoRA) models

Some Hugging Face repositories ship LoRA adapter weights applied on top of a
fal-hosted base model (the Hub marks them with `adapter: "lora"` in the
inference-provider mapping). fal only applies those weights when the request
carries a `loras` entry, so the provider injects

```json
{"loras": [{"path": "https://huggingface.co/<the HF model ID>/resolve/main/<weights path>", "scale": 1}]}
```

automatically for such models — matching huggingface.js. Without it the
request succeeds but silently runs the base model. Models mapped to fal's
generic SD-class LoRA endpoint (`fal-ai/lora`) additionally gain the
`model_name` entry fal requires for that endpoint, on text-to-image only:
unlike dedicated LoRA endpoints whose base model is baked in, the generic
endpoint must be told which base to load, and the Hub mapping carries no
base-model metadata, so the SDK defaults it to SDXL (the same constant both
official clients send). Pass your own `loras` or `model_name` via
`ProviderParams` to override either injected entry (unlike huggingface.js,
which overwrites caller-supplied values).

### Provider-specific request parameters

fal models take provider-native parameters, not the canonical Hugging Face
ones, and many endpoints reject unknown fields. The canonical request types
do not cover every knob fal models expose, so every request DTO carries an
advanced escape hatch for this:

```go
image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene mountain landscape at sunset",
	ProviderParams: map[string]any{
		"enable_safety_checker": true,
	},
})
```

`ProviderParams` entries are merged into the flat body after the typed
parameters, so entries override typed fields — including the codec-provided
`image_urls` array and the injected `loras` entry (last write wins). This
mirrors the `extra_body` argument of the official Python client. The one
exception is image-segmentation's `sync_mode`, which the provider forces
after `ProviderParams` so the mask always returns inline, exactly as the
official clients request it.

### Provider-specific response metadata

Responses may carry provider-specific details in `ProviderMetadata`
(e.g. `request_id`, `queue_position`, `inference_time`). Providers populate
the map only when there is something to add; it is `nil` otherwise.

Quick start with the Hub resolver:

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewFalAIProvider(resolver)
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("krea/Krea-2-Turbo"),
	hfopts.WithProvider(provider),
	hfopts.WithMaxResponseBodyBytes(16<<20),
)

image, err := client.GenerateImage(hftypes.TextToImageRequest{
	Input: "A serene mountain landscape at sunset",
})
```

Generated videos and images can exceed the SDK's default 1 MiB response cap,
so raise it with `hfopts.WithMaxResponseBodyBytes` when needed.

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own fal API key in your
Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
resolved endpoint stay unchanged — and your fal account is billed instead.
See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing its mixins

Like the Hugging Face provider, Fal AI is composed from per-task mixins:
`FalAITextToImageMixin`, `FalAIImageToImageMixin`, `FalAITextToVideoMixin`,
`FalAIImageSegmentationMixin`, and `FalAISpeechRecognitionMixin`. Unlike the
Hugging Face mixins, each carries an exported `Resolver ModelResolver` field
that maps the configured Hugging Face model ID to fal's identifier at
endpoint-construction time — embed the mixins for the tasks you serve and set
each `Resolver` explicitly (or use `NewFalAIProvider` to wire all five). The
resolved mapping is attached to the endpoint and the pipeline hands it to the
codec, so adapter-model payloads get their LoRA injection without a second
lookup; standalone codec use falls back to the codec's own resolver. Each
mixin serves an `AsyncCodec`: its codecs encode fal's flat body and complete
the submit-poll-fetch-download exchange themselves. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the pattern.

## See also

- [examples/fal-ai](../../examples/fal-ai) — runnable programs for each
  supported task, wired with the Hub resolver.
- [Fal AI on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/fal-ai)
- [architecture.md](../architecture.md)
