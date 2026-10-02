// Package hfopts defines the functional options used to configure hfgo
// clients and individual requests.
//
// Options are supplied to clients at construction time via hfgo.NewClient
// and may be overridden per request. The canonical Options type lives here,
// outside the root package, so the public API and the internal request
// plumbing can share one implementation without import cycles or re-exports.
package hfopts
