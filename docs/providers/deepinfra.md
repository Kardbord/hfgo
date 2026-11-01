# DeepInfra provider

`hfproviders.DeepInfraProvider` serves chat completion, speech recognition,
and pooled feature extraction through the Hugging Face router. DeepInfra is
the first built-in partner provider that mixes the two routing mechanisms the
provider layer offers: chat completions pin routing through the model-string
suffix (like the other chat partner providers), while the pipeline tasks
resolve DeepInfra's own model identifiers through a model resolver (like
WaveSpeed and Fal AI).

This page covers only what is specific to DeepInfra; the provider layer's
design is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                               | Endpoint method                 | Codec                             | Client method               |
| ---------------------------------- | -------------------------------- | ----------------------------------- | ---------------------------- |
| Chat completion (unary)            | `ChatEndpoint`                  | `ChatCodec`                       | `Chat`                       |
| Chat completion (streaming)        | `ChatEndpoint`                  | `ChatStreamCodec`                 | `ChatStream`                 |
| Speech recognition                 | `SpeechRecognitionEndpoint`     | `SpeechRecognitionCodec`          | `RecognizeSpeech`            |
| Feature extraction (pooled)        | `FeatureExtractionEndpoint`     | `FeatureExtractionPooledCodec`    | `ExtractPooledFeatures`      |
| Feature extraction (pooled, batch) | `FeatureExtractionBatchEndpoint`| `FeatureExtractionPooledBatchCodec`| `ExtractPooledFeaturesBatch` |

Structured message content — including the image parts vision-language models
take — passes through the shared chat codec unchanged, so VLM requests need no
extra wiring. DeepInfra's embeddings endpoint returns one vector per input and
never per-token embeddings, so the token-level feature-extraction interfaces
are not implemented; casting the provider to one of them — or to any other
pipeline-task interface it does not serve — fails with a configuration
`*hferrors.SDKError`, exactly as with the default provider.

DeepInfra also serves text-to-speech on the Hugging Face router; this SDK does
not implement that task yet, so the provider does not expose it.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and chat needs no
[model resolver](../architecture.md#external-provider-model-resolution): the
router performs provider mapping server-side.

The task pipeline appends `:deepinfra` to the resolved model string unless the
model already carries an explicit suffix, so

```go
provider, err := hfproviders.NewDeepInfraProvider(hfproviders.NewStaticModelResolver(nil))
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("openai/gpt-oss-120b"),
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

sends `"model": "openai/gpt-oss-120b:deepinfra"` on the wire. Chat never
consults the resolver, so the static resolver in the example (which fails
every lookup) is fine for chat-only use; a `HubModelResolver` is only required
for the pipeline tasks below.

The `:deepinfra` suffix is a provider pin, not a preference: the router routes
to DeepInfra deterministically. Hugging Face's automatic failover — which
applies to unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred`
policies — does not reroute an explicit provider: if DeepInfra does not serve
the model or task, the router returns an error rather than running the request
on another provider. A request model that already carries a suffix is passed
through as-is, which opts out of the pin.

> **Note:** the official Hugging Face clients route DeepInfra chat through
> the provider-scoped path (`deepinfra/v1/openai/chat/completions`) with a
> resolved model ID. This SDK routes through the unified endpoint with the
> suffix — the same convention as every other chat partner provider here.

## Model resolution (pipeline tasks)

Callers configure the Hugging Face model ID they know; the provider translates
it to DeepInfra's own model identifier at endpoint-construction time via a
[model resolver](../architecture.md#external-provider-model-resolution), and
the codecs reuse the resolution (the identifier rides the request body, not
the path). The constructor refuses a nil resolver, and the zero-value provider
fails endpoint resolution with a configuration `*hferrors.SDKError` rather
than panicking. Resolution is genuinely needed: some Hub mappings differ from
the DeepInfra identifier (e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b`
resolves to `nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`).

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
		Provider: "deepinfra",
		Task:     hfproviders.TaskAutomaticSpeechRecognition,
		Model:    "nvidia/nemotron-3.5-asr-streaming-0.6b",
	}: {ProviderModelID: "nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"},
})

provider, err := hfproviders.NewDeepInfraProvider(mapper)
if err != nil {
	log.Fatal(err)
}
```

## Speech recognition

DeepInfra's transcription endpoint (`deepinfra/v1/openai/audio/transcriptions`
on the router) accepts multipart form data only — unlike the Hugging Face
provider's raw-bytes or JSON variants. The codec encodes:

- the decoded audio on the `file` part, with its Content-Type sniffed from the
  bytes' magic signature and the part named `audio.<ext>` accordingly (`.wav`
  is the fallback for unrecognized formats, matching the official clients);
- the canonical `Parameters.ReturnTimestamps` as the `return_timestamps` form
  field, plus every `ProviderParams` entry (strings pass through, booleans and
  numbers stringify, other values JSON-encode);
- the mapped provider model identifier on the `model` field, applied last so
  `ProviderParams` entries cannot override it.

Responses decode from JSON: the transcript rides `text`, and when the response
carries `segments` they map onto canonical chunks with `[start, end]` second
timestamps (`resp.Chunks`). Every other top-level field (e.g. `usage`,
`duration`, `language`) rides `resp.ProviderMetadata`.

## Feature extraction

The embeddings endpoint (`deepinfra/v1/openai/embeddings` on the router)
speaks OpenAI's embeddings format. The codec sends `{"input": ..., "model":
<mapped id>}` — a string for single requests, an array for
`ExtractPooledFeaturesBatch` — and decodes `data[].embedding` into pooled
embeddings, one per input in order.

The hf-inference feature-extraction parameters the canonical request carries
(`normalize`, `prompt_name`, `truncate`, `truncation_direction`) are
hf-inference-specific and are **not forwarded** — DeepInfra's embeddings API
does not accept them. DeepInfra-specific parameters such as `encoding_format`
and `dimensions` ride the request's `ProviderParams` field instead:

```go
embedding, err := client.ExtractPooledFeatures(hftypes.FeatureExtractionRequest{
	Input: "What is the capital of France?",
	ProviderParams: map[string]any{
		"encoding_format": "float",
	},
})
```

`ProviderParams` entries merge into the body after `input`, and the mapped
`model` is applied last, so entries can never override it. This mirrors the
`extra_body` argument of the official Python client.

Quick start with the Hub resolver:

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewDeepInfraProvider(resolver)
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("BAAI/bge-base-en-v1.5"),
	hfopts.WithProvider(provider),
)

embedding, err := client.ExtractPooledFeatures(hftypes.FeatureExtractionRequest{
	Input: "What is the capital of France?",
})
```

## Billing

Routed requests are billed on your Hugging Face account only for providers
that are pay-as-you-go compatible. If you have HF credits but receive
`402 Payment Required` from this provider, add your own DeepInfra API key in
your Hugging Face account settings (Inference Providers section): the router
swaps the authentication transparently — the SDK code, `HF_TOKEN`, and the
`:deepinfra` suffix stay unchanged — and your DeepInfra account is billed
instead. See the
[pricing docs](https://huggingface.co/docs/inference-providers/pricing) for
the routed-versus-custom-key comparison.

## Reusing its mixins

Like the other providers, DeepInfra is composed from per-task mixins:
`HuggingFaceChatMixin` — the same zero-byte mixin the default provider and the
chat-only partners use — plus `DeepInfraSpeechRecognitionMixin` and
`DeepInfraFeatureExtractionMixin`, each carrying an exported
`Resolver ModelResolver` field that maps the configured Hugging Face model ID
to DeepInfra's identifier at endpoint-construction time. Embed the mixins for
the tasks you serve and set each `Resolver` explicitly (or use
`NewDeepInfraProvider` to wire them all). The resolved mapping is attached to
the endpoint and the pipeline hands it to the codec; standalone codec use
falls back to the codec's own resolver. The feature-extraction mixin serves
only the pooled interfaces. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the pattern.

## See also

- [examples/deepinfra](../../examples/deepinfra) — runnable programs for each
  supported task: chat in `basic`, `streaming`, and `conversation` variants,
  and the pipeline tasks wired with the Hub resolver.
- [DeepInfra on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/deepinfra)
- [Groq provider](./groq.md) — the chat-routing convention the chat half
  follows.
- [Fal AI provider](./fal-ai.md) — the resolver-backed pipeline-task pattern
  the other half follows.
- [architecture.md](../architecture.md)
