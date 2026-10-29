# hfgo

<p align="center">
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/build.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/unit-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/unit-tests.yml/badge.svg" alt="Unit Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/integration-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/lint.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/fuzz-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/fuzz-tests.yml/badge.svg" alt="Fuzz Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/codeql-analysis.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/codeql-analysis.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://codecov.io/gh/Kardbord/hfgo" ><img src="https://codecov.io/gh/Kardbord/hfgo/graph/badge.svg?token=TK2WOHFEVG" alt="Code Coverage"/></a>
  <a href="https://pkg.go.dev/github.com/Kardbord/hfgo/v4"><img src="https://pkg.go.dev/badge/github.com/Kardbord/hfgo/v4.svg" alt="Go Reference"></a>
  <a href="https://www.bestpractices.dev/projects/12720"><img src="https://www.bestpractices.dev/projects/12720/baseline" alt="OpenSSF Baseline"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/Kardbord/hfgo"><img alt="OpenSSF Scorecard" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.scorecard.dev%2Fprojects%2Fgithub.com%2FKardbord%2Fhfgo&query=%24.score&style=flat&logoSize=auto&label=openssf%20scorecard&color=%23591A99&link=https%3A%2F%2Fscorecard.dev%2Fviewer%2F%3Furi%3Dgithub.com%2FKardbord%2Fhfgo"></a>
  <a href="https://www.bestpractices.dev/projects/12720"><img src="https://www.bestpractices.dev/projects/12720/badge" alt="OpenSSF Best Practices"></a>
</p>

An unofficial Go SDK for the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index).
It provides type-safe client methods for every supported inference task and a
pluggable provider layer for alternate wire formats.

## ⚠️ v4 Release Candidate

**v4** is currently in release candidate status. The API may
evolve before the final v4.0.0 release. **v3** and earlier are deprecated
and no longer maintained. See [#72](https://github.com/Kardbord/hfgo/issues/72)
for more information.

## Requirements

- **Go 1.27 or later.**
- A Hugging Face API key. To get one, create a
  [Hugging Face](https://huggingface.co/) account and generate a
  [token](https://huggingface.co/settings/tokens). Examples and integration
  tests read the token from the `HF_TOKEN` environment variable.

## Quickstart

```bash
go get github.com/Kardbord/hfgo/v4@latest
```

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

func main() {
	client := hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel("deepseek-ai/DeepSeek-R1"),
	)

	response, err := client.Chat(hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role:    "user",
				Content: hftypes.ChatMessageContent{
					Text: new("Hello! What is the capital of France?"),
				},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	if len(response.Choices) > 0 && response.Choices[0].Message.Content != nil {
		fmt.Println(*response.Choices[0].Message.Content)
	}

	// Other tasks follow this shape. See examples/huggingface/speech-recognition,
	// examples/huggingface/text-classification, and the rest of examples/ for more.
}
```

Every task works this way: build a client, send a typed request, read a typed
response or error. The [examples](./examples) directory has a runnable program
for each task and variant.

### One client, many tasks

A client is not tied to a single task or model. The token, base URL, provider,
and HTTP configuration are shared, so the same client you use for speech
recognition is the client you use for text classification. Because each task
needs its own model, supply the model per call when it differs from the client
default:

```go
client := hfgo.NewClient(hfopts.WithToken(os.Getenv("HF_TOKEN")))

// Automatic speech recognition — model supplied per call.
transcript, err := client.RecognizeSpeech(
	hftypes.SpeechRecognitionRequest{Input: audioBase64},
	hfopts.WithModel("openai/whisper-large-v3"),
)

// Text classification — same client, different model.
labels, err := client.ClassifyText(
	hftypes.TextClassificationRequest{Input: "A delightful surprise."},
	hfopts.WithModel("ProsusAI/finbert"),
)
```

Alternatively, create the client with a model and override it per call only
where needed. See [Options precedence](./docs/architecture.md#options-precedence).

## Inference Tasks

Each link points to a runnable example. Most tasks have several variants
(`basic`, `params`, `batch`, `streaming`, `conversation`); see the
[examples README](./examples/README.md) for the convention.

- [Chat](./examples/huggingface/chat)
- [Feature Extraction](./examples/huggingface/feature-extraction)
- [Fill Mask](./examples/huggingface/fill-mask)
- [Image Classification](./examples/huggingface/image-classification)
- [Image Segmentation](./examples/huggingface/image-segmentation)
- [Object Detection](./examples/huggingface/object-detection)
- [Question Answering](./examples/huggingface/question-answering)
- [Speech Recognition](./examples/huggingface/speech-recognition)
- [Summarization](./examples/huggingface/summarization)
- [Table Question Answering](./examples/huggingface/table-question-answering)
- [Text Classification](./examples/huggingface/text-classification)
- [Text to Image](./examples/huggingface/text-to-image)
- [Token Classification](./examples/huggingface/token-classification)
- [Translation](./examples/huggingface/translation)
- [Zero-Shot Text Classification](./examples/huggingface/zero-shot-text-classification)

Partner providers have their own examples:

- [WaveSpeed](./examples/wavespeed) — text-to-video, text-to-image, and
  image-to-image through WaveSpeed.
- [Baseten](./examples/baseten) — chat completions (unary, streaming, and
  multi-turn conversation) pinned to Baseten.
- [Cerebras](./examples/cerebras) — chat completions (unary, streaming, and
  multi-turn conversation) pinned to Cerebras.

## Error Handling

Errors returned by client methods are one of two types in
[`hferrors`](./hferrors):

- **`APIError`** — a 4xx/5xx response from the API. Carries the status code,
  message, request ID, method, and URL. Helpers: `IsClientError()`,
  `IsServerError()`, `IsAuthenticationError()`, and `IsRateLimitError()`.
- **`SDKError`** — produced locally before or while decoding a response, or
  raised when the provider reports a failure (such as a failed remote task)
  inside an otherwise valid response. Carries a `SDKErrorKind` (`validation`,
  `configuration`, `serialization`, `transport`, `provider`, `internal`) and
  an optional wrapped error.

Use `errors.As` to inspect either type:

```go
var apiErr *hferrors.APIError
if errors.As(err, &apiErr) && apiErr.IsRateLimitError() {
	// back off and retry
}

var sdkErr *hferrors.SDKError
if errors.As(err, &sdkErr) {
	log.Printf("sdk %s error: %v", sdkErr.Kind, sdkErr.Err)
}
```

## Packages

| Package                                                  | Contents                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------- |
| [`hfgo`](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) | `Client`, `NewClient`, and inference endpoint methods                |
| [`hfopts`](./hfopts)                                     | `Options`, `Option`, and all `With*` helpers                         |
| [`hftypes`](./hftypes)                                   | Request, response, and parameter DTOs for every task                 |
| [`hferrors`](./hferrors)                                 | `APIError`, `SDKError`, and `SDKErrorKind`                           |
| [`hfproviders`](./hfproviders)                           | `Provider` interface, per-task provider interfaces, codecs, mixins   |
| [`hfraw`](./hfraw)                                       | Low-level HTTP/SSE escape hatch for endpoints the SDK does not model |
| [`hfgoversion`](./hfgoversion)                           | `Version` constant and `UserAgent()` helper                          |

See [docs/architecture.md](./docs/architecture.md) for design decisions.

## Providers

Requests are routed through a pluggable inference-provider layer (the
[`hfproviders`](./hfproviders) package). The default
[`HuggingFaceProvider`](https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders)
speaks the Hugging Face wire format directly, so for most use cases nothing
needs to be configured. Use `hfopts.WithProvider(p)` to set an explicit
provider on a client or a single call, and `hfopts.WithDefaultProvider()` to
use the default Hugging Face provider. On OpenAI-compatible endpoints (e.g.
chat completions), a model-string suffix chooses how the router picks a
provider: a provider name (e.g. `model:baseten`) routes to that provider
deterministically, while a routing policy (`:fastest`, `:cheapest`,
`:preferred`) lets the router choose, with automatic failover to another
provider. Without a suffix the default `:fastest` policy applies. See
<https://huggingface.co/docs/inference-providers/main/en/index>.

Built-in providers have their own guides:

- [Hugging Face provider](./docs/providers/huggingface.md) — the default; speaks
  the Hugging Face wire format directly.
- [Baseten provider](./docs/providers/baseten.md) — chat completions pinned to
  Baseten through the router's `:baseten` suffix.
- [Cerebras provider](./docs/providers/cerebras.md) — chat completions pinned to
  Cerebras through the router's `:cerebras` suffix.
- [WaveSpeed provider](./docs/providers/wavespeed.md) — text-to-video,
  text-to-image, and image-to-image through WaveSpeed.

Advanced users who want custom providers should implement the `Provider` interface
plus the per-task interfaces you want to support, as shown by the `EchoProvider`
example in the `hfproviders` package docs. The optional `HuggingFace*Mixin` and
`Wavespeed*Mixin` types exist as a convenience for reusing a built-in provider's
endpoint and codec defaults: embed the ones you want (for example
`HuggingFaceTextClassificationMixin` or `WavespeedTextToImageMixin`).
See [docs/architecture.md](./docs/architecture.md#per-task-provider-mixins).

## Documentation

- [API reference](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) — full godoc.
- [examples/](./examples) — runnable programs for every task and variant.
- [docs/](./docs) — documentation index, including
  [architecture](./docs/architecture.md),
  [release process](./docs/release-process.md), the
  [Bruno API collection](./docs/bruno.md), and provider guides under
  [docs/providers](./docs/providers).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md)
before opening a PR. Security issues should follow [SECURITY.md](./SECURITY.md).

## Releases

See the [GitHub Releases](https://github.com/Kardbord/hfgo/releases) page for
the release notes and published versions.

## License

MIT — see [LICENSE](./LICENSE).

## Additional Resources

- [Hugging Face](https://huggingface.co/)
- [Inference API JSON Schemas](https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks)
- [Model Hub](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index)
- [HF on GitHub](https://github.com/huggingface)
  - Official [Python SDK](https://github.com/huggingface/huggingface_hub)
  - Official [JavaScript SDK](https://github.com/huggingface/huggingface.js)
