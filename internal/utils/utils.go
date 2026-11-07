package utils

import (
	"context"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"time"
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

// ParseRetryAfter parses a Retry-After header expressed in seconds. It
// returns the delay capped at limit, or 0 when the header is absent, does not
// parse, or holds a non-positive value. A pathological value cannot stall a
// caller's retry loop longer than limit.
func ParseRetryAfter(h http.Header, limit time.Duration) time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(h.Get("Retry-After")))
	if err != nil || seconds <= 0 {
		return 0
	}

	// Compare in whole seconds before scaling: a value at or beyond the
	// limit saturates there, and the comparison also prevents the
	// seconds-to-duration multiplication from silently overflowing into a
	// negative (or bogus positive) duration.
	if int64(seconds) >= int64(limit/time.Second) {
		return limit
	}

	delay := time.Duration(seconds) * time.Second
	if delay > limit {
		return limit
	}

	return delay
}

// IsNil reports whether value is nil, including a nil pointer, map, slice,
// channel, or function stored inside a non-nil interface (a so-called typed
// nil). Interface fields checked only with == nil otherwise pass typed-nil
// values through, and calling a value-receiver method on them panics.
func IsNil(value any) bool {
	if value == nil {
		return true
	}

	reflected := reflect.ValueOf(value)
	if !reflected.IsValid() {
		return true
	}

	kind := reflected.Kind()
	nilable := kind == reflect.Pointer || kind == reflect.Map ||
		kind == reflect.Slice || kind == reflect.Chan ||
		kind == reflect.Func || kind == reflect.UnsafePointer

	return nilable && reflected.IsNil()
}
