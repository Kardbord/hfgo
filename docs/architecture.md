# hfgo Architecture

hfgo is an unofficial Go SDK for the
[Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index).
It provides type-safe client methods for every supported inference task and a
pluggable provider layer for alternate wire formats. The API surface favors
upstream feature parity; breaking changes are possible as the upstream API
evolves.

This document records the design decisions that shape the codebase. It is not
an API reference — see [pkg.go.dev](https://pkg.go.dev/github.com/Kardbord/hfgo/v4)
and `examples/` for usage documentation, and the code itself for full detail.

## Package Layout

| Package       | Import path                               | Contents                                                                                            |
| ------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `hfgo`        | `github.com/Kardbord/hfgo/v4`             | `Client`, `NewClient`, and inference endpoint methods                                               |
| `hfopts`      | `github.com/Kardbord/hfgo/v4/hfopts`      | `Options`, `Option`, and all `With*` helpers                                                        |
| `hftypes`     | `github.com/Kardbord/hfgo/v4/hftypes`     | Request/response DTOs for every task                                                                |
| `hferrors`    | `github.com/Kardbord/hfgo/v4/hferrors`    | `APIError`, `SDKError`, and `SDKErrorKind`                                                          |
| `hfproviders` | `github.com/Kardbord/hfgo/v4/hfproviders` | `Provider` interface, per-task `*Provider` interfaces, `Codec`, `AsProvider`, `HuggingFaceProvider` |
| `hfraw`       | `github.com/Kardbord/hfgo/v4/hfraw`       | Low-level HTTP/SSE escape hatch                                                                     |
| `hfgoversion` | `github.com/Kardbord/hfgo/v4/hfgoversion` | `Version` constant and `UserAgent()` helper                                                         |

Internal packages under `internal/` are implementation details and must only
be imported from within the module.

## Design Decisions

### Immutable client

A `Client` is an immutable value type that captures configuration at creation
time. Options are never mutated after construction. To change configuration,
create a new client. Each method call snapshots the client's options, so
concurrent calls are independent and deterministic.

A client is not bound to a task or model: the same client serves every method.
Because models are task-specific, supply the model per call with
`hfopts.WithModel(...)` when it differs from the client default; the client's
token, base URL, provider, and transport are reused unchanged.

Per-request options (variadic `hfopts.Option`) override client defaults for a
single call. Options are applied by value with defensive header copies;
contexts and HTTP clients are shared references.

### Options precedence

When a setting can be specified at multiple levels, the following precedence
applies (highest to lowest):

1. Request DTO fields (e.g. `ChatRequest.Model`)
2. Per-request options (e.g. `client.Chat(req, hfopts.WithModel(...))`)
3. Client-level options (e.g. `hfgo.NewClient(hfopts.WithModel(...))`)

This ensures more specific configuration always wins over more general
configuration.

### Request safety

Request DTOs are passed to Client methods **by value** and the SDK never
mutates the caller's payload. However, the value copy shares nested data
(slices, maps, pointed-to values) with the caller. The request and everything
it references must be treated as **read-only while a call is in flight**.

For concurrent invocation, pass a defensive copy per call
(`req.Clone()`) or build a fresh request per call. `Clone()` is a deep copy —
every slice gets new backing storage and every pointer a new pointee.

This is a deliberate trade-off: making DTOs non-copyable (mutexes, getters,
pointer-only APIs) would destroy the value-copy semantics the SDK is built on.
Preventing caller-authored races is therefore a caller responsibility.

### Concurrency model

Clients are safe for concurrent use out of the box. No explicit
synchronization is required. Concurrency safety is achieved through
immutability:

- Clients never mutate their options after creation.
- Each method call derives a fresh snapshot from the client's options.
- No shared mutable state between goroutines.

If you inject an HTTP client via `hfopts.WithHTTPClientFactory`, ensure it is
thread-safe by design or properly synchronized externally.

Stream objects are not clients. Typed streams such as `ChatStream` are not safe
for concurrent `Recv` calls. `Close` may be called concurrently with `Recv`.
Callers that need fan-out must distribute chunks from a single consumer loop.

### Provider abstraction

Requests are routed through a pluggable provider layer defined in the
`hfproviders` package. The Hugging Face Inference API is the reference wire
format, so the built-in `HuggingFaceProvider` is a no-op transform;
third-party providers can implement the same interfaces to serve HF-format
requests from their own wire formats.

A provider advertises task support structurally through per-task interfaces
(e.g. `ChatProvider`, `SummarizationProvider`). Each interface pairs a task's
endpoint-resolution method (`{Noun}Endpoint`) with its codec accessor
(`{Noun}Codec`). Dispatch casts via `AsProvider`, which returns a
configuration error when the cast fails.

**Security invariant**: endpoint methods return paths *relative* to
`Options.BaseURL`. The transport rejects absolute paths, so a provider can
never redirect a request (and its bearer token) to a host the caller did not
choose. Authentication is owned by the caller via `Options.Token` /
`Options.Headers`; providers and codecs must not set `Authorization`.

### Codecs

The `Codec` interface is the wire-format transform. `Encode` receives the
canonical request and returns the body plus request headers describing the
format; `Decode` receives the response body/headers and owns content-type
validation. This keeps task functions agnostic to wire format: providers own
endpoint construction and any request/response translation.

Codec-returned request headers are applied as *defaults* — caller-set headers
win. The exceptions: codec-returned `Authorization` values are dropped
(authentication is caller-owned), and streaming forces
`Accept: text/event-stream` (SSE framing is a transport concern).

When a codec or streaming decode hook returns an existing
`*hferrors.SDKError`, the SDK preserves that error and its `SDKErrorKind`
instead of relabeling it as a serialization failure. This keeps validation and
configuration failures distinguishable from malformed JSON.

### Naming convention

Client methods use an **action verb** (what the caller does); provider
interfaces, endpoints, codecs, and DTOs use the **upstream task noun**
(matching Hugging Face's task taxonomy). For example, the client method is
`ClassifyText` but the DTO is `TextClassificationRequest` and the provider
interface is `TextClassificationProvider`. There are no task-name constants —
task support is expressed structurally through the per-task interfaces.

### Error model

Errors are categorized into two types in the `hferrors` package:

- **`APIError`**: returned by the API (4xx/5xx HTTP responses). Carries the
  status code, message, body reader, request ID, method, and URL. Helper
  methods: `IsClientError()`, `IsServerError()`, `IsAuthenticationError()`,
  `IsRateLimitError()`.
- **`SDKError`**: produced locally (before an API response, or while
  unmarshaling one). Carries a `SDKErrorKind` category (`validation`,
  `configuration`, `serialization`, `transport`, `internal`), a message, and
  an optional underlying error.

### `hfraw` escape hatch

The `hfraw` package is the deliberate exception to the typed Client surface.
It exposes raw HTTP and SSE access to endpoints the SDK does not model
type-safely, with methods combining byte-slice or `io.Reader` bodies and
one-shot or SSE streaming. One-shot methods come in typed-error and
raw-response variants; streams always apply typed error handling. It is kept
separate to avoid cluttering the root `Client` API.

`Do` and the stream methods translate 4xx/5xx responses into
`*hferrors.APIError`, whereas `DoRaw` returns the raw `*http.Response` so
callers can inspect the status and headers themselves. Stream methods require
the response to advertise `text/event-stream`; a response that does not is
never handed to the SSE decoder, which would otherwise yield no events and
report `io.EOF`, silently dropping the response. A 4xx/5xx response that is not
an event stream is returned as an `*hferrors.APIError` whose `Body` the caller
owns, so the error payload is preserved; every other non-stream response is
returned as an `*hferrors.SDKError` and its body is drained and closed. The
status code and headers of a successfully opened stream are exposed via
`Stream.StatusCode` and `Stream.Header`.

### Chat streaming

`ChatStream` wraps an SSE stream of `ChatStreamResponse` chunks. It merges
tool-call metadata across deltas so each chunk includes complete tool-call
identifiers. Callers must `Close()` the stream to release the underlying HTTP
connection and decoder goroutine. Termination is final: once `io.EOF` is
observed, no further chunks are delivered.

A `ChatStream` is not safe for concurrent `Recv` calls. Its accumulator is
owned by the single consumer. `Close` may be called concurrently with `Recv`.

### Feature extraction output shapes

Feature extraction supports two model-dependent output families:

- **Pooled:** `ExtractFeatures` and `ExtractFeaturesBatch` return
  `FeatureExtraction` / `[]FeatureExtraction` (`[]float64` per input). Use these
  with sentence-transformers-style models for sentence embeddings, semantic
  search, and similarity.
- **Token-level:** `ExtractTokenFeatures` and `ExtractTokenFeaturesBatch`
  return `FeatureExtractionTokens` / `FeatureExtractionBatchTokens`
  (`[][]float64` for one input, `[][][]float64` for a batch). Use these with
  transformers-style models such as BART when you need per-token hidden states
  and will apply your own pooling.

The shape is determined by the model, not the request, so callers choose the
method that matches the model's output.

## Where to Make Changes

| Task                   | Location                                           |
| ---------------------- | -------------------------------------------------- |
| Request/response types | `hftypes/*` (no re-exports needed)                 |
| Endpoint behavior      | `internal/task/*` + `Client` method in `client.go` |
| Provider / wire format | `hfproviders/*`                                    |
| Options                | `hfopts/options.go`                                |
| Raw HTTP/SSE           | `hfraw/raw.go`                                     |
| Unit tests             | Next to the code they exercise                     |
| Integration tests      | `internal/integration_tests/*_integration_test.go` |
| Runnable examples      | `examples/<task>/<variant>/`                       |
