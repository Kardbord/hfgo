package hfproviders

import (
	"context"
	"net/http"
	"time"

	"github.com/Kardbord/hfgo/v4/internal/request"
)

// The async queue-based partner providers (WaveSpeed, fal.ai, Replicate)
// share one exchange shape: a submit call returns a provider task handle,
// follow-up polls read a status URL until the task reaches a terminal
// status, and provider-CDN outputs download unauthenticated. The exchange
// below carries the provider-independent half of one poll; each provider
// supplies its task type, its decode, and its terminal-status classification
// (see [queueExchange]).

// queueExchange binds the provider-specific steps of [queuePollOnce] to one
// task type: the transport predicates and Retry-After parsing the provider
// reuses across its exchange, plus the task-specific decode and
// terminal-status classification.
type queueExchange[T any] struct {
	// isContextError reports whether err is (or wraps) a context
	// cancellation or deadline expiry. Such errors are not transient: they
	// must surface immediately instead of being retried.
	isContextError func(error) bool

	// transientStatus reports whether an HTTP status is a transient poll
	// failure worth retrying (rate limiting and server errors).
	transientStatus func(int) bool

	// retryAfter parses a server-provided Retry-After header in seconds,
	// returning the wait override for the next round. Zero keeps the
	// regular polling schedule.
	retryAfter func(http.Header) time.Duration

	// decode turns a poll response body into the provider's task type.
	decode func([]byte) (T, error)

	// terminal classifies a decoded task. done reports whether the task
	// reached a terminal status: true with a nil error means it completed
	// (out holds its data); true with an error means it failed (err
	// describes the failure); false means it is still queued or running,
	// so polling should continue. A nil out is never returned with a nil
	// error.
	terminal func(*T) (*T, bool, error)
}

// queuePollOnce issues one poll request and classifies the result. out is
// non-nil when the task reached a terminal status; retryAfter is a
// server-provided wait override for the next round; any error stops polling.
func queuePollOnce[T any](
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	path string,
	exchange queueExchange[T],
) (*T, time.Duration, error) {
	//nolint:bodyclose // Drained and closed on transient and success branches;
	// non-transient 4xx/5xx errors transfer the body to the returned APIError,
	// which the caller must close per the APIError contract.
	resp, err := request.DoRaw(ctx, pollParams, http.MethodGet, path, nil)
	if err != nil {
		// Transport-level failures (dropped connections, timeouts) are
		// transient: keep polling until the context expires. Context
		// cancellation and deadline expiry are not transient: they must
		// surface immediately.
		if exchange.isContextError(err) {
			return nil, 0, err
		}

		return nil, 0, nil
	}

	if exchange.transientStatus(resp.StatusCode) {
		override := exchange.retryAfter(resp.Header)
		request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

		return nil, override, nil
	}
	// Non-transient error: transfer the open body to the returned APIError
	// so the caller can inspect it per the APIError contract.
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, 0, request.APIErrorFromResponse(resp, http.MethodGet)
	}

	body, readErr := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)
	if readErr != nil {
		return nil, 0, readErr
	}

	data, decodeErr := exchange.decode(body)
	if decodeErr != nil {
		return nil, 0, decodeErr
	}

	out, done, terminalErr := exchange.terminal(&data)
	if done {
		return out, 0, terminalErr
	}

	// Still queued or running: keep polling on the regular schedule.
	return nil, 0, nil
}
