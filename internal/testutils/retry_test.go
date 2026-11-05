//go:build !integration

package testutils

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/stretchr/testify/require"
)

func TestIsTransient(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       bool
	}{
		{"500 is transient", http.StatusInternalServerError, true},
		{"502 is transient", http.StatusBadGateway, true},
		{"503 is transient", http.StatusServiceUnavailable, true},
		{"504 is transient", http.StatusGatewayTimeout, true},
		{"529 is transient", statusServerOverloaded, true},
		{"429 is transient", http.StatusTooManyRequests, true},
		{"410 is never transient", http.StatusGone, false},
		{"401 is not transient", http.StatusUnauthorized, false},
		{"404 is not transient", http.StatusNotFound, false},
		{"400 is not transient", http.StatusBadRequest, false},
		{"501 is not transient", http.StatusNotImplemented, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := &hferrors.APIError{StatusCode: tt.statusCode}
			require.Equal(t, tt.want, IsTransient(err), "IsTransient(%d)", tt.statusCode)
		})
	}

	t.Run("non-API errors are not transient", func(t *testing.T) {
		t.Parallel()

		require.False(t, IsTransient(nil))
		require.False(t, IsTransient(&hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "bad request",
		}))
		require.False(t, IsTransient(io.EOF))
	})
}

// compressSchedule shrinks the retry schedule so tests do not sleep for
// seconds, and restores the production values when the test ends. Tests
// using it must not run in parallel because they mutate the package knobs.
func compressSchedule(t *testing.T) {
	t.Helper()

	oldRetries, oldBase, oldFallback, oldCap :=
		transientMaxRetries, transientBaseDelay, transient429FallbackDelay, transient429MaxDelay

	t.Cleanup(func() {
		transientMaxRetries, transientBaseDelay, transient429FallbackDelay, transient429MaxDelay =
			oldRetries, oldBase, oldFallback, oldCap
	})

	transientMaxRetries = 3
	transientBaseDelay = time.Millisecond
	transient429FallbackDelay = time.Millisecond
	transient429MaxDelay = 2 * time.Millisecond
}

// opFor returns an op that performs a request against the given server and
// translates non-2xx responses into an *hferrors.APIError the same way the
// SDK's transport layer does, keeping the response headers.
func opFor(serverURL string) func() (string, error) {
	return func() (string, error) {
		resp, err := http.Get(serverURL) //nolint:gosec,noctx // test helper: fixed test-server URL
		if err != nil {
			return "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindTransport,
				Message: "request failed",
				Err:     err,
			}
		}
		defer resp.Body.Close()

		if resp.StatusCode >= http.StatusBadRequest {
			return "", &hferrors.APIError{
				StatusCode: resp.StatusCode,
				Message:    resp.Status,
				Headers:    resp.Header.Clone(),
			}
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "read failed",
				Err:     err,
			}
		}

		return string(body), nil
	}
}

func TestRetryTransient_SucceedsAfterTransient503s(t *testing.T) {
	compressSchedule(t)

	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		if hits <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)

			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	t.Cleanup(server.Close)

	value := RetryTransient(context.Background(), t, opFor(server.URL))
	require.Equal(t, "ok", value)
	require.Equal(t, 3, hits, "expected two 503 retries before success")
}

func TestRetryTransient_Persistent503FailsAfterBoundedAttempts(t *testing.T) {
	compressSchedule(t)

	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	_, err := retryTransientErr(context.Background(), t, opFor(server.URL))
	require.Error(t, err, "persistent 503 must not succeed")
	require.Equal(t, 4, hits, "expected 1 attempt + 3 retries")
}

func TestRetryTransient_410FailsImmediately(t *testing.T) {
	compressSchedule(t)

	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.WriteHeader(http.StatusGone)
	}))
	t.Cleanup(server.Close)

	_, err := retryTransientErr(context.Background(), t, opFor(server.URL))
	require.Error(t, err, "410 must not succeed")

	var apiErr *hferrors.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, http.StatusGone, apiErr.StatusCode)
	require.Equal(t, 1, hits, "410 must never be retried")
}

func TestRetryTransient_429Retries(t *testing.T) {
	t.Run("without Retry-After falls back to fixed delay", func(t *testing.T) {
		compressSchedule(t)

		var hits int
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hits++
			if hits == 1 {
				w.WriteHeader(http.StatusTooManyRequests)

				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		}))
		t.Cleanup(server.Close)

		value := RetryTransient(context.Background(), t, opFor(server.URL))
		require.Equal(t, "ok", value)
		require.Equal(t, 2, hits)
	})

	t.Run("transientWait honors server Retry-After", func(t *testing.T) {
		err := &hferrors.APIError{
			StatusCode: http.StatusTooManyRequests,
			Headers:    http.Header{"Retry-After": []string{"1"}},
		}

		wait, waitErr := transientWait(err, 1)
		require.NoError(t, waitErr)

		// The raw delay is 1s; symmetric jitter of up to 25% bounds the wait
		// to [0.75s, 1.25s]. The header delay must be honored rather than
		// replaced by the fixed fallback.
		require.GreaterOrEqual(t, wait, 750*time.Millisecond)
		require.LessOrEqual(t, wait, 1250*time.Millisecond)
	})

	t.Run("transientWait 429 without header uses fallback", func(t *testing.T) {
		err := &hferrors.APIError{StatusCode: http.StatusTooManyRequests}

		wait, waitErr := transientWait(err, 1)
		require.NoError(t, waitErr)

		lower := transient429FallbackDelay * 3 / 4
		upper := transient429FallbackDelay * 5 / 4
		require.GreaterOrEqual(t, wait, lower)
		require.LessOrEqual(t, wait, upper)
	})

	t.Run("transientWait caps server delay", func(t *testing.T) {
		err := &hferrors.APIError{
			StatusCode: http.StatusTooManyRequests,
			Headers:    http.Header{"Retry-After": []string{"3600"}},
		}

		wait, waitErr := transientWait(err, 1)
		require.NoError(t, waitErr)
		require.Equal(t, transient429MaxDelay, wait, "pathological Retry-After must be capped")
	})
}

func TestRetryTransient_TransientWaitBackoffSchedule(t *testing.T) {
	t.Parallel()

	err := &hferrors.APIError{StatusCode: http.StatusServiceUnavailable}

	for attempt, base := range []time.Duration{
		transientBaseDelay,
		2 * transientBaseDelay,
		4 * transientBaseDelay,
	} {
		wait, waitErr := transientWait(err, attempt+1)
		require.NoError(t, waitErr)

		lower := base * 3 / 4
		upper := base * 5 / 4
		require.GreaterOrEqual(t, wait, lower, "attempt %d below jitter floor", attempt+1)
		require.LessOrEqual(t, wait, upper, "attempt %d above jitter ceiling", attempt+1)
	}
}

func TestRetryTransient_ContextCancelledStopsRetrying(t *testing.T) {
	compressSchedule(t)

	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := retryTransientErr(ctx, t, opFor(server.URL))
	require.Error(t, err)
	require.Equal(t, 1, hits, "cancelled context must prevent any retry")
}

func TestRetryTransient_TypePreserved(t *testing.T) {
	compressSchedule(t)

	type payload struct {
		Value int
	}

	value := RetryTransient(context.Background(), t, func() (payload, error) {
		return payload{Value: 42}, nil
	})
	require.Equal(t, payload{Value: 42}, value)
}
