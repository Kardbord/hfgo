// Package hfopts defines the functional options used to configure hfgo clients
// and individual requests.
//
// Options are supplied to a client at construction time and may be overridden
// per request. The canonical [Options] type lives here, outside the root package,
// so the public API and the internal request plumbing can share one
// implementation without import cycles or re-exports.
//
// Common options include [WithToken], [WithModel], [WithBaseURL], [WithContext],
// [WithHeader], and [WithProvider]. Use [WithDefaultProvider] and
// [WithDefaultHTTPClient] to restore the built-in defaults.
//
// # Precedence
//
// A per-request [Option] overrides the same setting on the client for that call
// only; values that are not overridden fall back to the client's configuration.
// See the architecture notes for details:
// https://github.com/Kardbord/hfgo/blob/main/docs/architecture.md#options-precedence
package hfopts
