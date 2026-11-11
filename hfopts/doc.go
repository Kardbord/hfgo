// Package hfopts defines the functional options used to configure hfgo clients
// and individual requests.
//
// Options are supplied to a client at construction time and may be overridden
// per request. The canonical [Options] type lives here, outside the root package,
// so the public API and the internal request plumbing can share one
// implementation without import cycles or re-exports.
//
// Common options include [WithToken], [WithModel], [WithBaseURL],
// [WithHeader], and [WithProvider]. Use [WithDefaultProvider] and
// [WithDefaultHTTPClient] to restore the built-in defaults.
//
// Request contexts are not options: every client method takes a
// context.Context as its first argument, so deadlines and cancellation
// signals are scoped to a single call rather than stored in configuration.
//
// # Precedence
//
// A per-request [Option] overrides the same setting on the client for that call
// only; values that are not overridden fall back to the client's configuration.
package hfopts
