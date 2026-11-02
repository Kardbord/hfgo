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
| [providers/novita.md](./providers/novita.md)         | The Novita provider: chat-only task coverage, `:novita` model routing, and how to reuse the chat mixin.                                                                                                    |
| [providers/ovhcloud.md](./providers/ovhcloud.md)     | The OVHcloud provider: chat-only task coverage, `:ovhcloud` model routing, and how to reuse the chat mixin.                                                                                                |
| [providers/publicai.md](./providers/publicai.md)     | The Public AI provider: chat-only task coverage, `:publicai` model routing, and how to reuse the chat mixin.                                                                                                |
| [providers/zai-org.md](./providers/zai-org.md)       | The Z.ai provider: chat-only task coverage, `:zai-org` model routing, and how to reuse the chat mixin.                                                                                                     |
| [providers/wavespeed.md](./providers/wavespeed.md)     | The WaveSpeed provider: model resolution, supported tasks, and how to reuse its mixins.                                                                                                                    |
| [providers/fal-ai.md](./providers/fal-ai.md)           | The Fal AI provider: model resolution, supported tasks, the queue exchange, and how to reuse its mixins.                                                                                                   |
| [providers/deepinfra.md](./providers/deepinfra.md)   | The DeepInfra provider: chat pinning via the `:deepinfra` suffix, resolver-backed speech recognition and feature extraction, and how to reuse its mixins.                                                        |
| [providers/together.md](./providers/together.md)     | The Together provider: chat pinning via the `:together` suffix, including the codec that rewrites `json_schema` response formats to Together's flattened shape.                   |
| [providers/nscale.md](./providers/nscale.md)         | The Nscale provider: chat pinning via the `:nscale` suffix, resolver-backed text-to-image, and how to reuse its mixins.                                                                                           |
| [providers/scaleway.md](./providers/scaleway.md)     | The Scaleway provider: chat pinning via the `:scaleway` suffix, resolver-backed feature extraction, and how to reuse its mixins.                                                                                  |
| [providers/replicate.md](./providers/replicate.md)   | The Replicate provider: model resolution, supported tasks, the prediction exchange, and how to reuse its mixins.                                                                                                 |
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
