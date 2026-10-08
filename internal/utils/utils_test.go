//go:build !integration

package utils

import (
	"context"
	"net/http"
	"testing"

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
		base["x-test"], //nolint:staticcheck // non-canonical key is intentional
	)
	require.Empty(t, base.Get("X-Other"))
	require.Len(
		t,
		base["x-test"], //nolint:staticcheck // non-canonical key is intentional
		2,
		"original slice must not be reused",
	)
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
			base["x-replace"], //nolint:staticcheck // non-canonical source key is intentional
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
