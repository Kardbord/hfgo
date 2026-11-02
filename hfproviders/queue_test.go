//go:build !integration

package hfproviders

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// TestQueueRetryAfter exercises the shared Retry-After parser as a table:
// seconds are honored, garbage and non-positive values are ignored, and the
// cap bounds pathological values.
func TestQueueRetryAfter(t *testing.T) {
	t.Parallel()

	cases := []struct {
		header string
		want   time.Duration
	}{
		{header: "3", want: 3 * time.Second},
		{header: " 3 ", want: 3 * time.Second},
		{header: "soon", want: 0},
		{header: "0", want: 0},
		{header: "-3", want: 0},
		{header: "999999", want: queueMaxRetryAfter},
		{header: "", want: 0},
	}

	for _, tc := range cases {
		headers := http.Header{}
		if tc.header != "" {
			headers.Set("Retry-After", tc.header)
		}

		require.Equal(t, tc.want, queueRetryAfter(headers), "Retry-After %q", tc.header)
	}
}

// TestQueueTransientStatus exercises the shared transient-status predicate:
// rate limiting and server errors are transient, everything else is not.
func TestQueueTransientStatus(t *testing.T) {
	t.Parallel()

	require.True(t, queueTransientStatus(http.StatusTooManyRequests))
	require.True(t, queueTransientStatus(http.StatusInternalServerError))
	require.True(t, queueTransientStatus(http.StatusBadGateway))
	require.False(t, queueTransientStatus(http.StatusOK))
	require.False(t, queueTransientStatus(http.StatusBadRequest))
	require.False(t, queueTransientStatus(http.StatusNotFound))
}

// TestQueueIsContextError exercises the shared context-error predicate:
// context cancellation and deadline expiry count, including when wrapped;
// other errors do not.
func TestQueueIsContextError(t *testing.T) {
	t.Parallel()

	require.True(t, queueIsContextError(context.Canceled))
	require.True(t, queueIsContextError(context.DeadlineExceeded))
	require.True(t, queueIsContextError(fmt.Errorf("wrapped: %w", context.Canceled)))
	require.False(t, queueIsContextError(errors.New("connection reset")))
}

// TestQueuePollOnce_TransientNoOverride exercises the shared poll exchange's
// transport predicate wiring: a transient poll failure yields no task, no
// error, and a zero wait override when the server sends no Retry-After.
func TestQueuePollOnce_TransientNoOverride(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	pollParams := request.Params{
		HTTPClient: server.Client(),
		BaseURL:    server.URL,
		UserAgent:  "test-agent/1.0",
		Headers:    http.Header{},
	}

	out, override, err := queuePollOnce(
		context.Background(), PostSubmitParams{}, pollParams, "/task",
		queueExchange[string]{
			decode:   func(b []byte) (string, error) { return string(b), nil },
			terminal: func(*string) (*string, bool, error) { return nil, true, nil },
		},
	)
	require.NoError(t, err)
	require.Nil(t, out)
	require.Equal(t, time.Duration(0), override)
}

// TestQueueDecodeMediaBody exercises the shared output-media validation: a
// matching advertised type is kept verbatim, a generic or mismatched type is
// byte-sniffed and accepted when the sniff agrees with the prefix, and the
// advertised type is rejected only when the bytes agree with it or cannot be
// identified at all.
func TestQueueDecodeMediaBody(t *testing.T) {
	t.Parallel()

	// pngBody is the full PNG file signature, which http.DetectContentType
	// reports as image/png.
	pngBody := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	t.Run("matching advertised type is kept", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"image/png"}},
		}

		body, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, pngBody, body)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("generic advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"application/octet-stream"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("mismatched advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"text/plain"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("unparseable advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"%%%"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("body contradicting the advertised type is rejected", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"foo":"bar"}`),
			Headers: http.Header{"Content-Type": {"application/octet-stream"}},
		}

		_, _, err := queueDecodeMediaBody(params, "image/")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "image/")
		require.ErrorContains(t, err, "application/octet-stream")
	})

	t.Run("unidentifiable body is rejected", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    []byte("not an image"),
			Headers: http.Header{"Content-Type": {"text/plain"}},
		}

		_, _, err := queueDecodeMediaBody(params, "image/")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "text/plain")
	})
}
