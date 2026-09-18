# hfgo

<p align="center">
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/build.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/unit-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/unit-tests.yml/badge.svg" alt="Unit Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/integration-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/integration-tests.yml/badge.svg" alt="Integration Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/lint.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/fuzz-tests.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/fuzz-tests.yml/badge.svg" alt="Fuzz Tests"></a>
  <a href="https://github.com/Kardbord/hfgo/actions/workflows/codeql-analysis.yml"><img src="https://github.com/Kardbord/hfgo/actions/workflows/codeql-analysis.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://pkg.go.dev/github.com/Kardbord/hfgo/v4"><img src="https://pkg.go.dev/badge/github.com/Kardbord/hfgo/v4.svg" alt="Go Reference"></a>
  <a href="https://www.bestpractices.dev/projects/12720"><img src="https://www.bestpractices.dev/projects/12720/baseline" alt="OpenSSF Baseline"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/Kardbord/hfgo"><img alt="OpenSSF Scorecard" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.scorecard.dev%2Fprojects%2Fgithub.com%2FKardbord%2Fhfgo&query=%24.score&style=flat&logoSize=auto&label=openssf%20scorecard&color=%23591A99&link=https%3A%2F%2Fscorecard.dev%2Fviewer%2F%3Furi%3Dgithub.com%2FKardbord%2Fhfgo"></a>
  <a href="https://www.bestpractices.dev/projects/12720"><img src="https://www.bestpractices.dev/projects/12720/badge" alt="OpenSSF Best Practices"></a>
</p>

An unofficial Go SDK for the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index).
Directly call any model available in the [Model Hub](https://huggingface.co/models).

An API key is required for authorized access. To get one, create a [Hugging Face](https://huggingface.co/) account and generate a [token](https://huggingface.co/settings/tokens).

## ⚠️ v4 Release Candidate

**v4** is currently in release candidate status (v4.0.0-rc1). The API may
evolve before the final v4.0.0 release. **v3** and earlier are deprecated
and no longer maintained. See [#72](https://github.com/Kardbord/hfgo/issues/72)
for more information.

## Usage

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
)

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	client := hfgo.NewClient(
		hfgo.WithToken(token),
		hfgo.WithModel("deepseek-ai/DeepSeek-R1"),
	)

	request := hfgo.ChatRequest{
		Messages: []hfgo.ChatMessage{
			{
				Role: "user",
				Content: hfgo.ChatMessageContent{
					Text: Ptr("Hello! What is the capital of France?"),
				},
			},
		},
		MaxTokens: Ptr(1024),
	}

	response, err := client.Chat(request)
	if err != nil {
		log.Fatalf("Failed to complete chat request: %v", err)
	}

	for _, choice := range response.Choices {
		if choice.Message.Content != nil {
			fmt.Println(*choice.Message.Content)
		}
	}
}

func Ptr[T any](v T) *T {
	return &v
}
```

See the [examples](./examples) directory for more.

## Inference Tasks

- [Chat](./examples/chat)
- [Fill Mask](./examples/fill-mask)
- [Question Answering](./examples/question-answering)
- [Summarization](./examples/summarization)
- [Speech Recognition](./examples/speech-recognition)
- [Table Question Answering](./examples/table-question-answering)
- [Text Classification](./examples/text-classification)
- [Token Classification](./examples/token-classification)
- [Translation](./examples/translation)
- [Zero-Shot Text Classification](./examples/zero-shot-text-classification)

## Contributing

Contributions are welcome in many forms!

- **Opening or commenting on issues** to suggest new features, clarify requirements, and report bugs
- **Reviewing PRs** to help improve code quality
- **Documentation improvements** (updating README, docs/, or examples/)
- **Community engagement** (helping new contributors, answering questions)

If you plan to contribute code, please open an issue first to discuss your proposed changes, coordinate with maintainers, and avoid duplicate work.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

## Resources

- [Hugging Face](https://huggingface.co/)
- [Inference API JSON Schemas](https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks)
- [Model Hub](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index)
- [HF on GitHub](https://github.com/huggingface)
  - Official [Python SDK](https://github.com/huggingface/huggingface_hub)
  - Official [JavaScript SDK](https://github.com/huggingface/huggingface.js)
