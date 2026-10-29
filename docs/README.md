# hfgo Documentation

This directory holds longer-form documentation. For install, quickstart, and
the task list, start with the [main README](../README.md).

| Document                                             | What it covers                                                                                                                                                                                             |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](./architecture.md)                 | Design decisions: immutable `hfgo` clients, options precedence, request safety and concurrency, the provider/codec abstraction, Hub model resolution and its internal client, the error model, and the `hfraw` escape hatch. Also maps common change locations. |
| [providers/huggingface.md](./providers/huggingface.md) | The default Hugging Face provider: supported tasks, how chat provider suffixes work, and how to reuse its mixins.                                                                                          |
| [providers/baseten.md](./providers/baseten.md)         | The Baseten provider: chat-only task coverage, `:baseten` model routing, and how to reuse the chat mixin.                                                                                                  |
| [providers/cerebras.md](./providers/cerebras.md)       | The Cerebras provider: chat-only task coverage, `:cerebras` model routing, and how to reuse the chat mixin.                                                                                                |
| [providers/cohere.md](./providers/cohere.md)         | The Cohere provider: chat-only task coverage, `:cohere` model routing, and how to reuse the chat mixin.                                                                                                    |
| [providers/featherless-ai.md](./providers/featherless-ai.md) | The Featherless AI provider: chat-only task coverage, `:featherless-ai` model routing, and how to reuse the chat mixin.                                                                            |
| [providers/fireworks-ai.md](./providers/fireworks-ai.md) | The Fireworks provider: chat-only task coverage, `:fireworks-ai` model routing, and how to reuse the chat mixin.                                                                                       |
| [providers/groq.md](./providers/groq.md)             | The Groq provider: chat-only task coverage, `:groq` model routing, and how to reuse the chat mixin.                                                                                                        |
| [providers/ovhcloud.md](./providers/ovhcloud.md)     | The OVHcloud provider: chat-only task coverage, `:ovhcloud` model routing, and how to reuse the chat mixin.                                                                                                |
| [providers/wavespeed.md](./providers/wavespeed.md)     | The WaveSpeed provider: model resolution, supported tasks, and how to reuse its mixins.                                                                                                                    |
| [release-process.md](./release-process.md)           | How releases are cut with release-please, including the release-candidate workflow for draft branches.                                                                                                     |
| [bruno.md](./bruno.md)                               | The [Bruno](https://www.usebruno.com/) collection for manually exercising the API without writing Go code. Primarily for SDK development.                                                                  |

## Other places to look

- [examples/](../examples) — a runnable program per task and variant; see the
  [examples README](../examples/README.md) for the naming convention and the
  provider-specific examples.
- [pkg.go.dev](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) — the full API
  reference.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development setup, build/test
  commands, and PR conventions.
