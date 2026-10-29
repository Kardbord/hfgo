# Examples

Runnable programs demonstrating each inference task. Examples are organized by
provider: `examples/<provider>/<task>/<variant>`. The default Hugging Face
provider's examples live under [`huggingface/`](./huggingface), partner-provider
examples under their provider's directory (e.g. [`wavespeed/`](./wavespeed)), and
[`custom-provider`](./custom-provider) shows how to plug in a fictional
provider. Each `main` package is self-contained. Per-task examples read their
token from the `HF_TOKEN` environment variable, with the exception of
[`huggingface/chat/openai-compatibility`](./huggingface/chat/openai-compatibility),
which necessarily reads its token from `OPENROUTER_TOKEN`. The
[`custom-provider`](./custom-provider) example makes no inference call and
needs no token.

Each program constructs its own client for clarity, but that is not required:
one client can be reused across every task — only the model changes. See
[One client, many tasks](../README.md#one-client-many-tasks).

> **Cost notice:** Most examples call the live Hugging Face Inference API and
> may incur charges on your account. The same applies to the integration tests
> and the [Bruno collection](../tools/bruno/hfgo/). The `custom-provider`
> example only resolves an endpoint and sends no request. The WaveSpeed
> `hub-resolver` examples additionally query the Hub's model mapping data;
> that metadata lookup is not an inference call.

## Running an example

```bash
HF_TOKEN=hf_... go run ./examples/huggingface/chat/basic
```

Or set the variable once for the session:

```bash
export HF_TOKEN=hf_...
go run ./examples/huggingface/feature-extraction/batch
```

## Variant convention

Variant directories are named for what they add over the `basic` example:

| Variant             | Meaning                                                                            |
| ------------------- | ---------------------------------------------------------------------------------- |
| `basic`             | The smallest complete call for the task. Start here.                               |
| `params`            | Passes a `*Parameters` struct to tune the request (e.g. `TopK`, `Temperature`).    |
| `batch`             | Sends multiple inputs in one request. Only for tasks the API supports batching on. |
| `streaming`         | Consumes a server-sent-events stream. Currently chat only.                         |
| `conversation`      | Maintains multi-turn message history. Currently chat only.                         |
| `token-level`       | Returns token-level embeddings instead of pooled vectors. Feature extraction only. |
| `token-level-batch` | Returns token-level embeddings for multiple inputs. Feature extraction only.       |

Not every task has every variant. WaveSpeed examples are the exception to this
naming scheme: their variants are named for the model resolver they wire (see
[WaveSpeed](#wavespeed)). The full matrix:

| Task                                                             | Variants                                                       |
| ---------------------------------------------------------------- | -------------------------------------------------------------- |
| [Chat](./huggingface/chat)                                       | `basic`, `streaming`, `conversation`, `openai-compatibility`   |
| [Feature Extraction](./huggingface/feature-extraction)           | `basic`, `params`, `batch`, `token-level`, `token-level-batch` |
| [Fill Mask](./huggingface/fill-mask)                             | `basic`, `params`                                              |
| [Image Classification](./huggingface/image-classification)       | `basic`, `params`                                              |
| [Image Segmentation](./huggingface/image-segmentation)           | `basic`, `params`                                              |
| [Object Detection](./huggingface/object-detection)               | `basic`, `params`                                              |
| [Question Answering](./huggingface/question-answering)           | `basic`, `params`                                              |
| [Speech Recognition](./huggingface/speech-recognition)           | `basic`, `params`                                              |
| [Summarization](./huggingface/summarization)                     | `basic`, `params`                                              |
| [Table Question Answering](./huggingface/table-question-answering) | `basic`, `params`                                            |
| [Text Classification](./huggingface/text-classification)         | `basic`                                                        |
| [Text to Image](./huggingface/text-to-image)                     | `basic`, `params`                                              |
| [Token Classification](./huggingface/token-classification)       | `basic`, `params`                                              |
| [Translation](./huggingface/translation)                         | `basic`, `params`                                              |
| [Zero-Shot Text Classification](./huggingface/zero-shot-text-classification) | `basic`                                            |

### WaveSpeed

The [WaveSpeed provider](../docs/providers/wavespeed.md) serves text-to-video,
text-to-image, and image-to-image through the Hugging Face router. Its examples
each show the provider and resolver wiring the default provider does not need.
The model resolver is the piece that varies per deployment, so WaveSpeed
variants are named for the resolver they wire: `static-resolver` pins the
HF-to-WaveSpeed mapping in code (no Hub lookup), while `hub-resolver` resolves
against the Hub's `inferenceProviderMapping` data live.

| Task                                          | Variants                          |
| --------------------------------------------- | --------------------------------- |
| [Text to Video](./wavespeed/text-to-video)    | `static-resolver`, `hub-resolver` |
| [Text to Image](./wavespeed/text-to-image)    | `static-resolver`, `hub-resolver` |
| [Image to Image](./wavespeed/image-to-image)  | `static-resolver`, `hub-resolver` |

### Baseten

The [Baseten provider](../docs/providers/baseten.md) pins chat completions to
Baseten: the chat endpoint and codecs are the default provider's, and the
provider appends the router's `:baseten` suffix to the model. Its variants
follow the standard naming scheme:

| Task                   | Variants                             |
| ---------------------- | ------------------------------------ |
| [Chat](./baseten/chat) | `basic`, `streaming`, `conversation` |

### Cerebras

The [Cerebras provider](../docs/providers/cerebras.md) pins chat completions to
Cerebras: the chat endpoint and codecs are the default provider's, and the
provider appends the router's `:cerebras` suffix to the model. Its variants
follow the standard naming scheme:

| Task                     | Variants                             |
| ------------------------ | ------------------------------------ |
| [Chat](./cerebras/chat)  | `basic`, `streaming`, `conversation` |

### Cohere

The [Cohere provider](../docs/providers/cohere.md) pins chat completions to
Cohere: the chat endpoint and codecs are the default provider's, and the
provider appends the router's `:cohere` suffix to the model. Its variants
follow the standard naming scheme:

| Task                   | Variants                             |
| ---------------------- | ------------------------------------ |
| [Chat](./cohere/chat)  | `basic`, `streaming`, `conversation` |

### Featherless AI

The [Featherless AI provider](../docs/providers/featherless-ai.md) pins chat
completions to Featherless AI: the chat endpoint and codecs are the default
provider's, and the provider appends the router's `:featherless-ai` suffix to
the model. Its variants follow the standard naming scheme:

| Task                          | Variants                             |
| ----------------------------- | ------------------------------------ |
| [Chat](./featherless-ai/chat) | `basic`, `streaming`, `conversation` |

### Fireworks

The [Fireworks provider](../docs/providers/fireworks-ai.md) pins chat
completions to Fireworks: the chat endpoint and codecs are the default
provider's, and the provider appends the router's `:fireworks-ai` suffix to
the model. Its variants follow the standard naming scheme:

| Task                          | Variants                             |
| ----------------------------- | ------------------------------------ |
| [Chat](./fireworks-ai/chat)   | `basic`, `streaming`, `conversation` |

### Groq

The [Groq provider](../docs/providers/groq.md) pins chat completions to
Groq: the chat endpoint and codecs are the default provider's, and the
provider appends the router's `:groq` suffix to the model. Its variants
follow the standard naming scheme:

| Task                 | Variants                             |
| -------------------- | ------------------------------------ |
| [Chat](./groq/chat)  | `basic`, `streaming`, `conversation` |

In addition to per-task examples, [custom-provider](./custom-provider) shows
how to plug in a fictional `FooProvider` that maps Hugging Face model IDs to
its own model identifiers through an explicitly injected model resolver,
without sending an inference call.

## See also

- [Main README](../README.md) — install and quickstart.
- [docs/architecture.md](../docs/architecture.md) — design decisions behind the
  client and provider layers.
- [pkg.go.dev](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) — full API
  reference, including `Example*` functions that render on the docs site.
