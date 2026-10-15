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
// # Concurrency
//
//   - A [Client] value is immutable and safe for concurrent use.
//   - Callers must not mutate request bodies after they are dispatched or while
//     they may be in flight.
//
// [hfgo.Client]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4#Client
package hfraw
