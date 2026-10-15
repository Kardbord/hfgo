# Examples

Runnable programs demonstrating each inference task, one directory per task and
one subdirectory per variant. Each `main` package is self-contained and reads
its token from the `HF_TOKEN` environment variable.

Each program constructs its own client for clarity, but that is not required:
one client can be reused across every task — only the model changes. See
[One client, many tasks](../README.md#one-client-many-tasks).

> **Cost notice:** These examples call the live Hugging Face Inference API and
> may incur charges on your account. The same applies to the integration tests
> and the [Bruno collection](../tools/bruno/hfgo/).

## Running an example

```bash
HF_TOKEN=hf_... go run ./examples/chat/basic
```

Or set the variable once for the session:

```bash
export HF_TOKEN=hf_...
go run ./examples/feature-extraction/batch
```

## Variant convention

Variant directories are named for what they add over the `basic` example:

| Variant        | Meaning                                                                            |
| -------------- | ---------------------------------------------------------------------------------- |
| `basic`        | The smallest complete call for the task. Start here.                               |
| `params`       | Passes a `*Parameters` struct to tune the request (e.g. `TopK`, `Temperature`).    |
| `batch`        | Sends multiple inputs in one request. Only for tasks the API supports batching on. |
| `streaming`    | Consumes a server-sent-events stream. Currently chat only.                         |
| `conversation` | Maintains multi-turn message history. Currently chat only.                         |

Not every task has every variant. The full matrix:

| Task                                                             | Variants                             |
| ---------------------------------------------------------------- | ------------------------------------ |
| [Chat](./chat)                                                   | `basic`, `streaming`, `conversation` |
| [Feature Extraction](./feature-extraction)                       | `basic`, `params`, `batch`           |
| [Fill Mask](./fill-mask)                                         | `basic`, `params`                    |
| [Image Classification](./image-classification)                   | `basic`, `params`                    |
| [Image Segmentation](./image-segmentation)                       | `basic`, `params`                    |
| [Object Detection](./object-detection)                           | `basic`, `params`                    |
| [Question Answering](./question-answering)                       | `basic`, `params`                    |
| [Speech Recognition](./speech-recognition)                       | `basic`, `params`                    |
| [Summarization](./summarization)                                 | `basic`, `params`                    |
| [Table Question Answering](./table-question-answering)           | `basic`, `params`                    |
| [Text Classification](./text-classification)                     | `basic`                              |
| [Text to Image](./text-to-image)                                 | `basic`, `params`                    |
| [Token Classification](./token-classification)                   | `basic`, `params`                    |
| [Translation](./translation)                                     | `basic`, `params`                    |
| [Zero-Shot Text Classification](./zero-shot-text-classification) | `basic`                              |

## See also

- [Main README](../README.md) — install and quickstart.
- [docs/architecture.md](../docs/architecture.md) — design decisions behind the
  client and provider layers.
- [pkg.go.dev](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) — full API
  reference, including `Example*` functions that render on the docs site.
