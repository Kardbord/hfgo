//go:build !integration

package hub

import (
	"container/list"
	"context"
	"errors"
	"testing"

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
