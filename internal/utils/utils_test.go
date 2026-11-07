//go:build !integration

package utils

import (
	"context"
	"math"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type sample struct{}

func TestIsNil(t *testing.T) {
	t.Parallel()

	var nilAny any
	var nilPtr *sample
	var nilMap map[string]string
	var nilSlice []string
	var nilChan chan int
	var nilFunc func()
	var nilIface error

	require.True(t, IsNil(nilAny))
	require.True(t, IsNil(nilPtr))
	require.True(t, IsNil(nilMap))
	require.True(t, IsNil(nilSlice))
	require.True(t, IsNil(nilChan))
	require.True(t, IsNil(nilFunc))
	require.True(t, IsNil(nilIface))

	// Typed nils wrapped in interfaces are still nil.
	var asAny any = nilPtr
	var asError error = (*sampleError)(nil)
	require.True(t, IsNil(asAny))
	require.True(t, IsNil(asError))

	require.False(t, IsNil(sample{}))
	require.False(t, IsNil(&sample{}))
	require.False(t, IsNil(""))
}

type sampleError struct{}

func (*sampleError) Error() string { return "boom" }

func TestNormalizeContext(t *testing.T) {
	t.Parallel()

	var nilCtx context.Context
	require.Equal(t, context.Background(), NormalizeContext(nilCtx))

	type ctxKey struct{}
	ctx := context.WithValue(context.Background(), ctxKey{}, "value")
	require.Equal(t, ctx, NormalizeContext(ctx))
}

func TestCloneHeader(t *testing.T) {
	t.Parallel()

	require.Nil(t, CloneHeader(nil))

	base := http.Header{"x-test": []string{"one", "two"}}
	cloned := CloneHeader(base)

	require.Equal(t, []string{"one", "two"}, cloned.Values("X-Test"))
	require.Equal(t, "one", cloned.Get("X-Test"))

	cloned.Set("X-Test", "changed")
	cloned.Add("X-Other", "new")

	// The source header deliberately uses a non-canonical key to verify that
	// cloning preserves the caller's original map key.
	require.Equal(
		t,
		[]string{"one", "two"},
		base["x-test"], //nolint:staticcheck // SA1008: non-canonical key is intentional
	)
	require.Empty(t, base.Get("X-Other"))
	require.Len(
		t,
		base["x-test"], //nolint:staticcheck // SA1008: non-canonical key is intentional
		2,
		"original slice must not be reused",
	)
}

func TestParseRetryAfter(t *testing.T) {
	t.Parallel()

	const retryCap = 60 * time.Second

	tests := []struct {
		name   string
		header string
		want   time.Duration
	}{
		{"missing header", "", 0},
		{"valid seconds", "30", 30 * time.Second},
		{"padded whitespace", "  45  ", 45 * time.Second},
		{"zero is ignored", "0", 0},
		{"negative is ignored", "-5", 0},
		{"not an integer", "soon", 0},
		{"not a plain integer", "1.5", 0},
		{"one below cap", "59", 59 * time.Second},
		{"exactly at cap", "60", retryCap},
		{"overflow threshold", "9999999999", retryCap}, // just above MaxInt64/1e9
		{"overflow above", "10000000000", retryCap},    // multiply would overflow
		{"huge value", "999999999999999999", retryCap}, // large but parseable
		{"max int64", "9223372036854775807", retryCap}, // overflow range
		{"capped at max", "999999", retryCap},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			h := http.Header{}
			if tt.header != "" {
				h.Set("Retry-After", tt.header)
			}

			require.Equal(
				t,
				tt.want,
				ParseRetryAfter(h, retryCap),
				"ParseRetryAfter(%q)",
				tt.header,
			)
		})
	}

	t.Run("nil header", func(t *testing.T) {
		t.Parallel()

		require.Equal(t, time.Duration(0), ParseRetryAfter(nil, retryCap))
	})

	t.Run("unbounded max", func(t *testing.T) {
		t.Parallel()

		h := http.Header{"Retry-After": []string{"90"}}
		require.Equal(t, 90*time.Second, ParseRetryAfter(h, time.Hour))
	})

	t.Run("zero limit yields zero for any value", func(t *testing.T) {
		t.Parallel()

		h := http.Header{"Retry-After": []string{"999999999999999999"}}
		require.Equal(t, time.Duration(0), ParseRetryAfter(h, 0))
	})

	t.Run("overflow saturates instead of wrapping negative", func(t *testing.T) {
		t.Parallel()

		// Before the overflow guard, values large enough to overflow the
		// seconds-to-duration multiplication wrapped into a negative
		// duration that bypassed the cap.
		for _, seconds := range []string{"10000000000", "999999999999999999", "9223372036854775807"} {
			h := http.Header{"Retry-After": []string{seconds}}
			got := ParseRetryAfter(h, retryCap)
			require.GreaterOrEqual(t, got, time.Duration(0), "ParseRetryAfter(%q)", seconds)
			require.Equal(t, retryCap, got, "ParseRetryAfter(%q)", seconds)
		}
	})

	t.Run("fractional limit is not truncated", func(t *testing.T) {
		t.Parallel()

		const limit = 1500 * time.Millisecond

		// A whole-second comparison against a truncated limit would
		// saturate here at 1.5s; the exact delay is 1s.
		h := http.Header{"Retry-After": []string{"1"}}
		require.Equal(t, time.Second, ParseRetryAfter(h, limit))

		h = http.Header{"Retry-After": []string{"2"}}
		require.Equal(t, limit, ParseRetryAfter(h, limit))

		h = http.Header{"Retry-After": []string{"0"}}
		require.Equal(t, time.Duration(0), ParseRetryAfter(h, limit))
	})

	t.Run("threshold boundary pair saturates identically", func(t *testing.T) {
		t.Parallel()

		// Exactly MaxInt64/time.Second seconds multiply without overflow
		// and saturate via the delay>limit clamp; one more second would
		// overflow and saturates via the pre-multiply guard. Both must
		// return the limit; keep the pair so the boundary cannot be
		// simplified away. The values are derived from the type bounds so
		// they always track the exact threshold.
		atThreshold := strconv.FormatInt(math.MaxInt64/int64(time.Second), 10)
		aboveThreshold := strconv.FormatInt(math.MaxInt64/int64(time.Second)+1, 10)
		for _, seconds := range []string{atThreshold, aboveThreshold} {
			h := http.Header{"Retry-After": []string{seconds}}
			require.Equal(t, retryCap, ParseRetryAfter(h, retryCap), "ParseRetryAfter(%q)", seconds)
		}
	})
}

func TestOverrideHeaders(t *testing.T) {
	t.Parallel()

	t.Run("nil base with override", func(t *testing.T) {
		t.Parallel()

		out := OverrideHeaders(nil, http.Header{"X-Test": []string{"yes"}})
		require.Equal(t, "yes", out.Get("X-Test"))
	})

	t.Run("nil base and nil override", func(t *testing.T) {
		t.Parallel()

		require.Nil(t, OverrideHeaders(nil, nil))
	})

	t.Run("merges and overrides canonicalized keys", func(t *testing.T) {
		t.Parallel()

		base := http.Header{"X-Keep": []string{"base"}, "x-replace": []string{"old"}}
		override := http.Header{"X-REPLACE": []string{"new"}, "X-New": []string{"added"}}

		out := OverrideHeaders(base, override)

		require.Equal(t, "base", out.Get("X-Keep"))
		require.Equal(t, "new", out.Get("X-Replace"))
		require.Equal(t, "added", out.Get("X-New"))
		require.Equal(
			t,
			[]string{"old"},
			base["x-replace"], //nolint:staticcheck // SA1008: non-canonical source key is intentional
			"base must not be mutated",
		)

		out.Set("X-Keep", "changed")
		require.Equal(t, "base", base.Get("X-Keep"))
	})
}

func TestEnsureHeader(t *testing.T) {
	t.Parallel()

	t.Run("nil header creates one", func(t *testing.T) {
		t.Parallel()

		out := EnsureHeader(nil, "X-Default", "value")
		require.Equal(t, "value", out.Get("X-Default"))
	})

	t.Run("existing value is preserved", func(t *testing.T) {
		t.Parallel()

		out := EnsureHeader(http.Header{"X-Test": []string{"existing"}}, "X-Test", "default")
		require.Equal(t, "existing", out.Get("X-Test"))
	})

	t.Run("empty value is replaced", func(t *testing.T) {
		t.Parallel()

		out := EnsureHeader(http.Header{"X-Test": []string{""}}, "X-Test", "default")
		require.Equal(t, "default", out.Get("X-Test"))
	})

	t.Run("missing key is set", func(t *testing.T) {
		t.Parallel()

		original := http.Header{"X-Existing": []string{"kept"}}
		out := EnsureHeader(original, "X-Default", "value")

		require.Equal(t, "value", out.Get("X-Default"))
		require.Equal(t, "kept", out.Get("X-Existing"))
		require.Empty(t, original.Get("X-Default"), "source header must not be mutated")
	})
}
