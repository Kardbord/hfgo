# Contributing to hfgo

Thank you for your interest in contributing! Contributions are welcome in
many forms: opening or commenting on issues, reviewing PRs, improving
documentation, and helping other community members.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We're
committed to providing a welcoming environment for all contributors.

## Getting Started

If you plan to contribute code, please open an issue first to discuss your
proposed changes and coordinate with maintainers.

### Prerequisites

- Go 1.27 or later
- Git
- [golangci-lint](https://golangci-lint.run/)
- Hugging Face API key (for integration tests only) — get one at
  <https://huggingface.co/settings/tokens> and set it as `HF_TOKEN`

### Setup

1. Fork the repository on GitHub.
2. Clone your fork and add the upstream remote:
   ```bash
   git clone https://github.com/your-username/hfgo.git
   cd hfgo
   git remote add upstream https://github.com/Kardbord/hfgo.git
   git fetch upstream
   ```
3. Create a feature branch:
   ```bash
   git checkout -b your-feature-or-fix upstream/main
   ```

## Building and Testing

Run the build script to format, lint, and test:

```bash
./tools/build.sh
```

Flags: `-f` adds fuzz tests, `-i` adds integration tests (may incur API
costs). The flags may be combined.

Or run individual commands:

```bash
gofmt -s -w .                        # Format
go mod tidy                          # Tidy dependencies
go vet ./...                         # Vet
golangci-lint run --fix ./...        # Lint
go build ./...                       # Build
go test -timeout 600s -v ./...       # Unit tests
go test -race -timeout 600s -v ./... # Race detection
```

## Repository Structure

See [docs/architecture.md](./docs/architecture.md) for the package layout and
design decisions, and [AGENTS.md](./AGENTS.md) for agent-oriented instructions.

## Pull Requests

1. Rebase on upstream main: `git fetch upstream && git rebase upstream/main`
2. Run all checks locally: `./tools/build.sh -fi`
3. Write or update tests for your changes.
4. Update documentation if needed.
5. Use a [Conventional Commits](https://www.conventionalcommits.org/) title
   (enforced by CI). Types: `feat`, `fix`, `refactor`, `docs`, `test`,
   `chore`. Mark breaking changes with `!` (e.g. `feat!: remove option`).
6. In the description, explain what changed and why; link related issues.

Code review etiquette: be responsive and polite. Make requested changes in
new commits (don't force push during review).

## Release Process

hfgo uses [release-please](https://github.com/googleapis/release-please) with
Conventional Commits in PR titles. See
**[docs/release-process.md](./docs/release-process.md)** for the full
description, including the release candidate workflow.

## Testing

### Build Tags

- **Unit tests** (`//go:build !integration`): run by default with
  `go test ./...`. No external services or API calls.
- **Integration tests** (`//go:build integration`, filename suffix
  `_integration_test.go`): run with `-tags=integration`. Require
  `HF_TOKEN` and hit the live API. Located in
  `internal/integration_tests/`.

On pull requests, integration tests require manual approval from a repository
member before they run (enforced via GitHub environment protection rules to
control API costs).

### Manual Testing with Bruno

The repo includes a [Bruno](https://www.usebruno.com/) collection in
`tools/bruno/hfgo/` for manually exercising the API. See
[docs/bruno.md](./docs/bruno.md).

## Documentation

- Document all exported functions, types, and fields with clear godoc
  comments. Explain the "why," not just the "what."
- Keep `README.md` and `docs/` in sync with code.
- Add godoc examples (`Example*` functions in `_test.go` files) for public
  API entry points.

## Code Style

The project uses **gofmt** for formatting and **golangci-lint** for linting.
All are run by `./tools/build.sh`.

## Concurrency and Safety

Clients are immutable value types and safe for concurrent use. Request DTOs
are passed by value and never mutated by the SDK. However, nested data is
shared with the caller — treat requests as read-only while a call is in
flight, and use `req.Clone()` for concurrent invocation.

When making changes:
- Don't introduce mutable state in clients.
- Test with `go test -race`.
- Document concurrency guarantees in comments.

## Common Mistakes to Avoid

1. **Mutating request data**: treat requests and their nested data as
   read-only while a call is in flight.
2. **Not closing streams**: always `Close()` `ChatStream` or `hfraw.Stream`.
3. **Sharing mutable HTTP clients**: ensure injected clients are thread-safe.
4. **Breaking API contracts**: changing function signatures is a breaking
   change.
5. **Ignoring context**: respect context cancellation and timeouts.

## Need Help?

Check existing issues, the [README](./README.md), [docs/](./docs/), and
[examples/](./examples/). Don't be afraid to open a new issue.

Thank you for contributing to hfgo!
