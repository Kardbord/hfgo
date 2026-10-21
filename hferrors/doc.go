// Package hferrors defines the error types returned by hfgo.
//
// Errors fall into two categories:
//
//   - [APIError] reports a 4xx/5xx response from the Hugging Face API. It carries
//     the status code, message, response body, and request ID, and classifies the
//     failure with helpers such as [APIError.IsClientError],
//     [APIError.IsServerError], [APIError.IsAuthenticationError], and
//     [APIError.IsRateLimitError].
//   - [SDKError] reports a local failure that occurs before a request is sent or
//     while a response is decoded. Its [SDKErrorKind] distinguishes configuration,
//     validation, serialization, transport, and internal failures.
//
// Use errors.As to extract either type.
//
// # Stream control
//
// A decoder, such as an [hfproviders.Codec], can steer a streaming pipeline by
// returning a control signal. [EndOfStreamError] ends the stream early, and
// [SkipEventError] suppresses a single frame. Both are recognized by errors.Is in
// value or pointer form. Unary pipelines treat any non-nil error as a failure.
//
// [hfproviders.Codec]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders#Codec
package hferrors
