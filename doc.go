// Package hfgo provides Go bindings for the Hugging Face Inference API.
//
// hfgo is an unofficial, type-safe client SDK for the Hugging Face Inference API.
// Create a [Client] with [NewClient], then use the [Client] methods to perform
// your desired inference task, such as [Client.Chat] or [Client.ClassifyText].
// Each method takes a typed request DTO from [hftypes] and returns a typed response
// or error. A single [Client] serves every task; only the model changes. Pass a
// per-request [hfopts.Option] such as [hfopts.WithModel] when a task needs a
// different model.
//
// # Status
//
// v4 is a release candidate. The API may change before v4.0.0, and earlier major
// versions are deprecated and no longer maintained. See the releases page for
// the latest version: https://github.com/Kardbord/hfgo/releases
//
// # Packages
//
//   - This package: the [Client] and one method per inference task.
//   - [hfopts]: client and per-request configuration, including [hfopts.Options].
//   - [hftypes]: request, response, and parameter DTOs for every task.
//   - [hferrors]: typed errors, APIError and SDKError.
//   - [hfproviders]: pluggable providers mapping a task to its wire format and endpoint.
//   - [hfraw]: raw HTTP and SSE access for endpoints the SDK does not model.
//   - [hfgoversion]: the SDK version and User-Agent string.
//
// # Client behavior
//
//   - A [Client] is immutable: options are fixed at construction and snapshotted
//     per call, so a client is safe for concurrent use by multiple goroutines.
//   - Per-request [hfopts.Option] values override client defaults for that call.
//   - Request DTOs are passed by value and the SDK never mutates the caller's
//     payload. Treat a request and the data it references as read-only while a
//     call is in flight; for concurrent calls, build a fresh request or pass a
//     copy from the DTO's Clone method.
//
// # Errors
//
// Methods return [hferrors.APIError] for non-2xx API responses and
// [hferrors.SDKError] for local failures. Inspect them with errors.As. For API
// responses, [hferrors.APIError] classifies the failure with helpers such as
// [hferrors.APIError.IsClientError] and [hferrors.APIError.IsRateLimitError].
// For local failures, [hferrors.SDKErrorKind] distinguishes configuration,
// validation, serialization, transport, and internal errors.
//
// # See also
//
//   - Runnable examples: https://github.com/Kardbord/hfgo/tree/main/examples
//   - Design decisions: https://github.com/Kardbord/hfgo/blob/main/docs/architecture.md
//
// [hferrors]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors
// [hferrors.APIError]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors#APIError
// [hferrors.APIError.IsClientError]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors#APIError.IsClientError
// [hferrors.APIError.IsRateLimitError]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors#APIError.IsRateLimitError
// [hferrors.SDKError]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors#SDKError
// [hferrors.SDKErrorKind]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hferrors#SDKErrorKind
// [hfproviders]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders
// [hfraw]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfraw
// [hfgoversion]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfgoversion
package hfgo
