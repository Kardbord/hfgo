//go:build !integration

package hub

import (
	"container/list"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/stretchr/testify/require"
)

func TestIsTransientFetchFailure(t *testing.T) {
	t.Parallel()

	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	tests := []struct {
		name string
		//nolint:containedctx // The table carries a per-case lookup context.
		ctx  context.Context
		err  error
		want bool
	}{
		{
			name: "unknown error is not transient",
			ctx:  context.Background(),
			err:  errors.New("boom"),
			want: false,
		},
		{
			name: "canceled context is not transient",
			ctx:  canceled,
			err:  &hferrors.SDKError{Kind: hferrors.SDKErrorKindTransport},
			want: false,
		},
		{
			name: "server error is transient",
			ctx:  context.Background(),
			err:  &hferrors.APIError{StatusCode: 500},
			want: true,
		},
		{
			name: "rate limit is transient",
			ctx:  context.Background(),
			err:  &hferrors.APIError{StatusCode: 429},
			want: true,
		},
		{
			name: "client error is not transient",
			ctx:  context.Background(),
			err:  &hferrors.APIError{StatusCode: 404},
			want: false,
		},
		{
			name: "transport SDK error is transient",
			ctx:  context.Background(),
			err:  &hferrors.SDKError{Kind: hferrors.SDKErrorKindTransport},
			want: true,
		},
		{
			name: "configuration SDK error is not transient",
			ctx:  context.Background(),
			err:  &hferrors.SDKError{Kind: hferrors.SDKErrorKindConfiguration},
			want: false,
		},
	}

	for i := range tests {
		tc := tests[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.want, isTransientFetchFailure(tc.ctx, tc.err))
		})
	}
}

func TestEvictLockedStopsWhenLRUIsEmpty(t *testing.T) {
	t.Parallel()

	client := NewClient(request.Params{}, Config{})
	// An inconsistent cache: more entries than the cap with no LRU nodes.
	// The guard must stop eviction rather than dereference a nil back node.
	client.maxEntries = 0
	client.entries = map[string]*cacheEntry{"orphan": {}}
	client.lru = list.New()

	require.NotPanics(t, func() {
		client.evictLocked()
	})
	require.Len(t, client.entries, 1)
}

func TestServeStale(t *testing.T) {
	t.Parallel()

	const model = "some/model"

	cached := []Mapping{{Provider: "cached"}}
	fresh := []Mapping{{Provider: "fresh"}}
	staleExpiry := time.Now().Add(-time.Minute)

	t.Run("returns cached when the entry was evicted", func(t *testing.T) {
		t.Parallel()

		client := NewClient(request.Params{}, Config{})
		observed := &cacheEntry{mappings: cached, expiry: staleExpiry}

		got := client.serveStale(model, observed, staleExpiry, cached)

		require.Equal(t, cached, got)
	})

	t.Run("prefers an entry inserted during the fetch", func(t *testing.T) {
		t.Parallel()

		client := NewClient(request.Params{}, Config{})
		current := &cacheEntry{
			mappings: fresh,
			expiry:   time.Now().Add(time.Minute),
			element:  client.lru.PushFront(model),
		}
		client.entries[model] = current
		observed := &cacheEntry{mappings: cached, expiry: staleExpiry}

		got := client.serveStale(model, observed, staleExpiry, cached)

		require.Equal(t, fresh, got)
	})

	t.Run("prefers an entry refreshed in place", func(t *testing.T) {
		t.Parallel()

		client := NewClient(request.Params{}, Config{})
		observed := &cacheEntry{
			mappings: cached,
			expiry:   staleExpiry,
			element:  client.lru.PushFront(model),
		}
		client.entries[model] = observed

		// Simulate a concurrent successful refresh of the same entry.
		observed.mappings = fresh
		observed.expiry = time.Now().Add(time.Minute)

		got := client.serveStale(model, observed, staleExpiry, cached)

		require.Equal(t, fresh, got)
	})

	t.Run("re-stamps the observed entry", func(t *testing.T) {
		t.Parallel()

		client := NewClient(request.Params{}, Config{CacheTTL: time.Minute})
		observed := &cacheEntry{
			mappings: cached,
			expiry:   staleExpiry,
			element:  client.lru.PushFront(model),
		}
		client.entries[model] = observed

		before := time.Now()
		got := client.serveStale(model, observed, staleExpiry, cached)

		require.Equal(t, cached, got)
		require.WithinDuration(
			t,
			before.Add(staleRetryInterval),
			observed.expiry,
			time.Second,
		)
	})

	t.Run("caps the grace period by the cache TTL", func(t *testing.T) {
		t.Parallel()

		const ttl = 10 * time.Millisecond

		client := NewClient(request.Params{}, Config{CacheTTL: ttl})
		observed := &cacheEntry{
			mappings: cached,
			expiry:   staleExpiry,
			element:  client.lru.PushFront(model),
		}
		client.entries[model] = observed

		before := time.Now()
		got := client.serveStale(model, observed, staleExpiry, cached)

		require.Equal(t, cached, got)
		require.WithinDuration(t, before.Add(ttl), observed.expiry, time.Second)
	})
}
