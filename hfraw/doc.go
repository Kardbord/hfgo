// Package hfraw provides the low-level escape hatch to the Hugging Face
// Inference API.
//
// It is the deliberate exception to the rest of the SDK. The root [hfgo.Client]
// exposes type-safe methods for everyday use; hfraw exposes raw HTTP and SSE
// primitives for endpoints the SDK does not model. Build a raw [Client] with
// [NewClient], then send a request with [Client.Do] or consume a server-sent
// event stream with [Stream.Recv].
//
// hfraw only routes requests; it does not validate request bodies or interpret
// upstream wire-format quirks. Callers are responsible for paths, bodies, query
// strings, and headers. Because it mirrors the underlying HTTP API, its surface
// is more likely to change as the upstream API evolves.
//
// # Streaming
//
// The stream methods require the response to be a server-sent event stream.
// The status code and headers of a stream response are available via
// [Stream.StatusCode] and [Stream.Header]. A 4xx/5xx response that is not
// an event stream is returned as a [hferrors.APIError] whose Body the caller
// must read and close; any other non-stream response is returned as a
// [hferrors.SDKError]. Surfacing these responses keeps an API error from being
// silently dropped when the body cannot be decoded as an event stream.
//
// # Concurrency
//
//   - A [Client] value is immutable and safe for concurrent use.
//   - Callers must not mutate request bodies after they are dispatched or while
//     they may be in flight.
//
// [hfgo.Client]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4#Client
package hfraw
