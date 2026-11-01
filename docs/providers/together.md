# Together provider

`hfproviders.TogetherProvider` serves chat completion and speech recognition
through the Hugging Face router. It mixes the two routing mechanisms the
provider layer offers: chat completions pin routing through the model-string
suffix (like the other chat partner providers), while speech recognition
resolves Together's own model identifiers through a model resolver (like
[DeepInfra](./deepinfra.md)). Together's chat requests also get a
provider-specific codec that adapts structured-output configuration to
Together's wire shape.

This page covers only what is specific to Together; the provider layer's
design is documented in
[architecture.md](../architecture.md#per-task-provider-mixins).

## Supported tasks

| Task                        | Endpoint method             | Codec                    | Client method     |
| --------------------------- | --------------------------- | ------------------------ | ----------------- |
| Chat completion (unary)     | `ChatEndpoint`              | `ChatCodec`              | `Chat`            |
| Chat completion (streaming) | `ChatEndpoint`              | `ChatStreamCodec`        | `ChatStream`      |
| Speech recognition          | `SpeechRecognitionEndpoint` | `SpeechRecognitionCodec` | `RecognizeSpeech` |

Structured message content — including the image parts vision-language models
take — passes through the chat codec unchanged, so VLM requests need no extra
wiring. Casting the provider to any pipeline-task interface it does not serve
(e.g. `TextToImageProvider`) fails with a configuration `*hferrors.SDKError`,
exactly as with the default provider.

Together also serves text-to-speech on the Hugging Face router, and the
official Hugging Face clients carry task classes for embeddings and image and
video generation; this SDK does not implement text-to-speech, and those other
tasks currently have no live Hub model mappings for Together, so the provider
does not expose them.

## Chat routing

Because chat payloads carry the model themselves, the endpoint is
model-independent (`POST v1/chat/completions`) and chat needs no
[model resolver](../architecture.md#external-provider-model-resolution): the
router performs provider mapping server-side.

The task pipeline appends `:together` to the resolved model string unless the
model already carries an explicit suffix, so

```go
provider, err := hfproviders.NewTogetherProvider(hfproviders.NewStaticModelResolver(nil))
if err != nil {
	log.Fatal(err)
}

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

sends `"model": "deepseek-ai/DeepSeek-V4.1-Flash:together"` on the wire. Chat
never consults the resolver, so the static resolver in the example (which
fails every lookup) is fine for chat-only use; a `HubModelResolver` is only
required for speech recognition.

The `:together` suffix is a provider pin, not a preference: the router routes
to Together deterministically. Hugging Face's automatic failover — which
applies to unsuffixed models and the `:fastest`, `:cheapest`, and `:preferred`
policies — does not reroute an explicit provider: if Together does not serve
the model or task, the router returns an error rather than running the request
on another provider. A request model that already carries a suffix is passed
through as-is, which opts out of the pin.

> **Note:** the official Hugging Face clients route Together chat through
> the provider-scoped path (`together/v1/chat/completions`) with a resolved
> model ID. This SDK routes through the unified endpoint with the suffix —
> the same convention as every other chat partner provider here.

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

Responses decode with the default provider's schema; Together omits
`system_fingerprint` from some responses and the canonical `hftypes` treat it
as optional, so no adaptation is needed.

## Model resolution (speech recognition)

Callers configure the Hugging Face model ID they know; the provider translates
it to Together's own model identifier at endpoint-construction time via a
[model resolver](../architecture.md#external-provider-model-resolution), and
the codec reuses the resolution (the identifier rides the request body, not
the path). The constructor refuses a nil resolver, and the zero-value provider
fails endpoint resolution with a configuration `*hferrors.SDKError` rather
than panicking. Hub mappings can rename models on the provider side at any
time, so resolving is the robust path even when the identifiers currently
match.

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
		Provider: "together",
		Task:     hfproviders.TaskAutomaticSpeechRecognition,
		Model:    "nvidia/nemotron-3.5-asr-streaming-0.6b",
	}: {ProviderModelID: "nvidia/nemotron-3.5-asr-streaming-0.6b"},
})

provider, err := hfproviders.NewTogetherProvider(mapper)
if err != nil {
	log.Fatal(err)
}
```

## Speech recognition

Together's transcription endpoint (`together/v1/audio/transcriptions` on the
router) accepts multipart form data only — unlike the Hugging Face provider's
raw-bytes or JSON variants. The codec encodes:

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

The canonical `SpeechRecognitionRequest.Input` carries base64-encoded audio
bytes. Together's endpoint additionally accepts a hosted HTTP(S) audio URL on
the `file` field — an official-clients extension this SDK does not model;
download the audio and pass its bytes if you need that input.

Quick start with the Hub resolver:

```go
resolver := hfproviders.NewHubModelResolver(
	hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
)

provider, err := hfproviders.NewTogetherProvider(resolver)
if err != nil {
	log.Fatal(err)
}

client := hfgo.NewClient(
	hfopts.WithToken(os.Getenv("HF_TOKEN")),
	hfopts.WithModel("nvidia/nemotron-3.5-asr-streaming-0.6b"),
	hfopts.WithProvider(provider),
)

resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
	Input: base64.StdEncoding.EncodeToString(audioBytes),
})
```

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

## Reusing its mixins

Together composes `HuggingFaceChatMixin` — the same zero-byte mixin the
default provider and the chat-only partners use — for the model-independent
chat endpoint, plus `TogetherSpeechRecognitionMixin`, which carries an
exported `Resolver ModelResolver` field that maps the configured Hugging Face
model ID to Together's identifier at endpoint-construction time. Embed the
speech-recognition mixin for a custom provider that serves only that task and
set `Resolver` directly (or use `NewTogetherProvider` to wire it). The
resolved mapping is attached to the endpoint and the pipeline hands it to the
codec; standalone codec use falls back to the codec's own resolver.

The chat codecs are deliberately *not* the mixin's plain JSON codecs: the
provider overrides `ChatCodec` and `ChatStreamCodec` to rewrite
`response_format` (see
[Structured outputs](#structured-outputs-json_schema-response-formats)). A
custom provider embedding only `HuggingFaceChatMixin` gets the default
envelope shape — the shape the official clients rewrite away, which is why
Together-bound structured output needs the special codec. The Together
codecs themselves are unexported: external providers needing the flattened
rewrite should embed `TogetherProvider` (overriding only the identity
methods) to inherit its codec accessors, or re-implement the rewrite over
`JSONCodec`. See
[Per-task provider mixins](../architecture.md#per-task-provider-mixins) for
the general pattern.

## See also

- [examples/together](../../examples/together) — runnable programs for each
  supported task: chat in `basic`, `streaming`, and `conversation` variants,
  and speech recognition wired with the Hub resolver.
- [Together on the Hugging Face Hub](https://huggingface.co/docs/inference-providers/providers/together)
- [Groq provider](./groq.md) — the chat-routing convention the chat half
  follows.
- [DeepInfra provider](./deepinfra.md) — the resolver-backed pipeline-task
  pattern the speech-recognition half follows.
- [architecture.md](../architecture.md)
