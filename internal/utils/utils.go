// Package utils provides small helpers shared across hfgo module packages
// for context normalization and defensive HTTP header manipulation.
package utils

import (
	"context"
	"net/http"
)

// NormalizeContext returns ctx when non-nil, otherwise context.Background.
func NormalizeContext(ctx context.Context) context.Context {
	if ctx != nil {
		return ctx
	}

	return context.Background()
}

// CloneHeader returns a deep copy of the provided headers with canonicalized keys.
func CloneHeader(h http.Header) http.Header {
	if h == nil {
		return nil
	}
	out := make(http.Header, len(h))
	for k, v := range h {
		out[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}

	return out
}

// OverrideHeaders copies base headers and replaces values with override entries.
func OverrideHeaders(base, override http.Header) http.Header {
	out := CloneHeader(base)
	if out == nil && len(override) > 0 {
		out = make(http.Header, len(override))
	}
	for k, v := range override {
		out[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}

	return out
}

// EnsureHeader returns a copy of headers with a default value set when missing or empty.
func EnsureHeader(h http.Header, key, value string) http.Header {
	out := CloneHeader(h)
	if out == nil {
		out = make(http.Header, 1)
	}
	if v := out.Get(key); v == "" {
		out.Set(key, value)
	}

	return out
}
