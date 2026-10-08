# Contributing to hfgo

Thank you for your interest in contributing to hfgo!

We welcome contributions in many forms:

- **Opening or commenting on issues** to clarify requirements or report bugs
- **Reviewing PRs** to help improve code quality
- **Documentation improvements** (updating README, docs/, or examples/)
- **Community engagement** (helping new contributors, answering questions)

This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.
We're committed to providing a welcoming environment for all contributors.

## Getting Started

If you plan to contribute code, please:

1. Open an issue first to discuss your proposed changes
2. Coordinate with maintainers to avoid duplicate work
3. Reference any relevant existing issues or discussions

### Prerequisites

- Go 1.27 or later
- Git
- Hugging Face API key (for running integration tests)
  - Get one at https://huggingface.co/settings/tokens
  - Set as `HF_TOKEN` environment variable
- golangci-lint

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/hfgo.git
   cd hfgo
   ```
3. Add the upstream remote to track the original repository:
   ```bash
   git remote add upstream https://github.com/Kardbord/hfgo.git
   ```
4. Verify your remotes are configured correctly:
   ```bash
   git remote -v
   ```
   You should see:
   ```
   origin    https://github.com/your-username/hfgo.git (fetch)
   origin    https://github.com/your-username/hfgo.git (push)
   upstream  https://github.com/Kardbord/hfgo.git (fetch)
   upstream  https://github.com/Kardbord/hfgo.git (push)
   ```
5. Fetch the latest changes from upstream:
   ```bash
   git fetch upstream
   ```
6. Create a feature branch from the latest upstream main:
   ```bash
   git checkout -b your-feature-or-fix upstream/main
   ```

### Building and Testing

Run the build script to format, lint, and test:

```bash
./tools/build.sh
```

Run the build script with `-i` to also include integration tests.
**Note**: Integration tests which will make API calls to Hugging Face
and will incur costs on your account.

```bash
./tools/build.sh -i

```
Or run individual commands:

```bash
# Format code
gofmt -s -w .

# Tidy dependencies
go mod tidy

# Vet code
go vet ./...

# Lint
golangci-lint run --fix ./...

# Build
go build ./...

# Unit tests
go test -timeout 600s -v ./...

# Race condition detection
go test -race -timeout 600s -v ./...

# Integration tests (requires HF_TOKEN)
go test -tags=integration -timeout 600s -v ./...
```

## Repository Structure

The repo is organized as a Go module with a small public root package
(`github.com/Kardbord/hfgo/v4`) and explicit public sub-packages for
different concerns (`hfopts`, `hftypes`, `hferrors`, `hfproviders`,
`hfgoversion`, `hfraw`). Implementation details live under `internal/`.

| Path | Purpose |
|------|---------|
| `client.go` | `Client`, `NewClient`, and every inference endpoint method |
| `hfraw/` | Low-level `Client`/`Stream`/`Event` escape hatch for arbitrary HTTP/SSE |
| `hfopts/` | `Options`, `Option`, and all `With*` option helpers |
| `hftypes/` | Request/response DTOs for every task (chat, fill-mask, classification, etc.) |
| `hferrors/` | `APIError`, `SDKError`, and `SDKErrorKind` |
| `hfproviders/` | Public `Provider` interface, per-task `*Provider` interfaces, `Codec` types (`JSONCodec` + HuggingFace built-ins), `AsProvider`, `HuggingFaceProvider` |
| `hfgoversion/` | `Version` constant and `UserAgent()` helper |
| `internal/task/` | Package-level task functions (`Chat`, `ClassifyText`, ...) |
| `internal/request/` | HTTP plumbing, JSON helpers, SSE parsing |
| `internal/utils/` | Shared helpers used by public and internal packages |
| `internal/testutils/` | Shared test helpers (mock transports, trackers) |
| `internal/integration_tests/` | Live-API integration tests |
| `examples/` | Runnable examples, one folder per task |
| `tools/bruno/` | Bruno collections for manual API testing |

**Where to make changes**:

- **Adding or modifying request/response types**: edit `hftypes/*`. Types are
  public directly from that package; there is no need to re-export them from
  the root package.
- **Adding or changing endpoint behavior**: edit the task function in
  `internal/task/*` and the corresponding `Client` method in `client.go`.
- **Provider or wire-format work**: edit `hfproviders/*`.
- **Tests**: unit tests live next to the code they exercise (e.g.
  `internal/task/*_task_test.go`); integration tests live in
  `internal/integration_tests/*_integration_test.go`.

## Pull Requests

### Before Creating a PR

1. Ensure your branch is up to date with main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run all checks locally:
   ```bash
   ./tools/build.sh -i
   ```

3. Write or update tests for your changes

4. Update documentation if needed

### Creating a PR

1. Push your feature branch to your fork
2. Create a PR on GitHub with:
   - **Title**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
     format (enforced by PR title check workflow)
     - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, etc.
     - Examples: `feat: add support for tool calling`,
       `fix: handle nil context in streaming`
     - Breaking changes: Use `feat!:` or `fix!:` to indicate breaking changes
   - **Description**: Explain what changes and why
   - **References**: Link to related issues (e.g., "Fixes #123")

### PR Title Examples

```
feat: add support for tool calling
fix: handle nil context in streaming
feat!: remove deprecated client option
docs: update API documentation
refactor: reorganize internal packages
test: add integration tests for streaming
```

### Code Review

- Be responsive to feedback
- Make requested changes in new commits (don't force push during review)
- Ask for clarification if feedback is unclear
- Be polite to reviewers (they should also be polite to you)

## Release Process

hfgo uses an **automated release workflow** driven by
[release-please](https://github.com/googleapis/release-please) and
[Conventional Commits](https://www.conventionalcommits.org/)
in the PR title.

See **[docs/release-process.md](./docs/release-process.md)** for the full
description, including release candidate (RC) workflow.

### How Mainline Releases Work

1. **PR is merged to `main`** with a conventional commit title.
2. **release-please analyzes commits** since the last release.
3. **Draft release PR is created** with:
   - Updated version in `hfgoversion/version.go`
   - Updated `.github/.release-please-manifest.json`
   - Auto-generated changelog
4. **Maintainer reviews and merges the release PR**.
   - The `validate-release-pr.yml` workflow verifies `go.mod` matches the
     release version.
5. **GitHub release is created automatically** with:
   - Git tag (e.g., v4.1.0)
   - Release notes with changelog

### Version Bumping

| Bump  | Trigger                                      |
|------ |----------------------------------------------|
| Major | Breaking change (`feat!:` or `fix!:` title)  |
| Minor | New feature (`feat:`)                        |
| Patch | Bug fix (`fix:`) or refactoring              |
| None  | Documentation, tests, chore, etc.            |

### Release Candidates

For major releases or significant changes, maintainers may create a
**release candidate (RC) branch** (e.g., `v5-draft`). See
[docs/release-process.md](./docs/release-process.md#release-candidate-rc-releases)
for details.

### Your Role in Releases

As a contributor, your role is:

- ✅ Write PR titles following Conventional Commits format
- ✅ Mark breaking changes with `!:` if applicable
- ✅ Ensure PR title accurately describes your changes

Maintainers handle:

- ✅ Reviewing Release PRs
- ✅ Merging Release PRs
- ✅ Managing RC branches and formal releases

## Testing

### Build Tags

Test files must include appropriate Go build tags to ensure they
run in the correct context:

- **Unit Tests**: Add `//go:build !integration` at the top of the file
  - These tests run by default with `go test ./...`
  - Should not require external services or API calls

- **Integration Tests**: Add `//go:build integration` at the top of the file
  - These tests only run with `-tags=integration`
  - Require a valid Hugging Face API token
  - Should be marked with the `_integration_test.go` filename suffix

**Example unit test**:

```go
//go:build !integration

package hfgo

import "testing"

func TestSomething(t *testing.T) {
    // Test code
}
```

**Example integration test**:

```go
//go:build integration

package integration_tests

import (
	"testing"

	"github.com/Kardbord/hfgo/v4"
)

func TestIntegration(t *testing.T) {
	// Integration test code; calls hfgo.Client directly against the live API
}
```

Integration tests live in `internal/integration_tests/` (package
`integration_tests`) and end with the `_integration_test.go` suffix so the
build tooling can discover them.

### Unit Tests

All code changes should include tests:

```bash
go test -timeout 600s -v ./...
```

### Integration Tests

Integration tests require a valid Hugging Face API token:

```bash
HF_TOKEN=your_token go test -tags=integration -timeout 600s -v ./...
```

On pull requests, integration tests require manual approval from a
repository member before they run. This is enforced via a
GitHub environment protection rule to control API costs. When a PR is
opened, the "Run Integration Tests" check will show as
"Waiting: Environment protection rules" until an authorized reviewer
approves the run in the GitHub Actions UI.

### Manual Testing with Bruno

The repo includes a [Bruno](https://www.usebruno.com/) collection in
`tools/bruno/hfgo/` for manually exercising the HuggingFace Inference API
endpoints without writing Go code. These requests hit the live API and may
incur costs, just like integration tests.

See **[docs/bruno.md](./docs/bruno.md)** and **[tools/bruno/hfgo/README.md](./tools/bruno/hfgo/README.md)** for setup and usage.

### Race Condition Detection

Always run race detection before submitting:

```bash
go test -race -timeout 600s -v ./...
```

## Documentation

### Code Comments

- Document all exported functions, types, and fields
- Use clear, concise language
- Explain the "why" not just the "what"
- Include examples for complex functions

### README and Docs

- Update README.md if adding new features
- Update examples/ if adding new features
- Keep documentation in sync with code, both in comments and in [docs/](./docs/)

## Code Style

The project uses:

- **gofmt** for formatting
- **golangci-lint** for linting rules

All of these are run by `./tools/build.sh`.

## Concurrency and Safety

The SDK prioritizes **concurrency safety**:

- Clients are immutable value types
- Client methods snapshot the client's options on each call
- Request DTOs are passed by value and never mutated by the SDK;
  callers prevent caller-side races (for example via each DTO's
  deep `Clone()` before concurrent invocation)
- No shared mutable state between goroutines
- All public APIs are safe for concurrent use

When making changes:

- Don't introduce mutable state in clients
- Test with `go test -race` to catch race conditions
- Document concurrency guarantees in comments

## Common Mistakes to Avoid

1. **Mutating request data**: Requests are passed to client methods **by
   value** and the SDK never mutates the caller's payload. However, the value
   copy shares nested data (slices, maps, and pointed-to values) with the
   caller, so treat the request and the data it references as **read-only
   while a call is in flight**. For concurrent invocation, pass a defensive
   copy per call, e.g. `client.Chat(req.Clone(), ...)`, or build a fresh
   request per call.
2. **Not closing streams**: Always call `Close()` on `ChatStream` or `hfraw.Stream`
3. **Sharing mutable HTTP clients**: If injecting HTTP clients, ensure thread-safety
4. **Breaking API contracts**: Changing function signatures is a breaking change
5. **Ignoring context**: Always respect context cancellation and timeouts

## Need Help?

- Check existing issues and discussions
- Read the [README.md](./README.md) and [docs](./docs/)
- Review examples in the [examples](./examples/) directory
- Don't be afraid to open a new issue

## Recognition

Contributors are recognized in:

- GitHub contributor graph
- Release notes (via commit attribution)

Thank you for contributing to hfgo! 🙏
