# hfgo Documentation

This directory holds longer-form documentation. For install, quickstart, and
the task list, start with the [main README](../README.md).

| Document | What it covers |
|----------|----------------|
| [architecture.md](./architecture.md) | Design decisions: immutable clients, options precedence, request safety and concurrency, the provider/codec abstraction, the error model, and the `hfraw` escape hatch. Also maps common change locations. |
| [release-process.md](./release-process.md) | How releases are cut with release-please, including the release-candidate workflow for draft branches. |
| [bruno.md](./bruno.md) | The [Bruno](https://www.usebruno.com/) collection for manually exercising the API without writing Go code. Primarily for SDK development. |

## Other places to look

- [examples/](../examples) — a runnable program per task and variant; see the
  [examples README](../examples/README.md) for the naming convention.
- [pkg.go.dev](https://pkg.go.dev/github.com/Kardbord/hfgo/v4) — the full API
  reference.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development setup, build/test
  commands, and PR conventions.
