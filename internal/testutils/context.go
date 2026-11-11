package testutils

import "context"

// NilContext returns a nil context value and exists solely so tests can
// exercise nil-context code paths without tripping staticcheck SA1012.
func NilContext() context.Context {
	return nil
}
