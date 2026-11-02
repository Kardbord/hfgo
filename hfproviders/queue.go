package hfproviders

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// The async queue-based partner providers (WaveSpeed, fal.ai, Replicate)
// share one exchange shape: a submit call returns a provider task handle,
// follow-up polls read a status URL until the task reaches a terminal
// status, and provider-CDN outputs download unauthenticated. The transport
// predicates and Retry-After parsing below are identical across those
// providers, so they live here once; the exchange carries the
// provider-specific half of one poll — its task type, its decode, and its
// terminal-status classification (see [queueExchange]).

// queueMaxRetryAfter caps a server-provided Retry-After so a pathological
// header cannot stall the polling loop for an unbounded time.
const queueMaxRetryAfter = 60 * time.Second

// queueIsContextError reports whether err is (or wraps) a context
// cancellation or deadline expiry. Such errors are not transient: they must
// stop polling immediately rather than be retried.
func queueIsContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// queueTransientStatus reports whether an HTTP status is a transient poll
// failure worth retrying (rate limiting and server errors).
func queueTransientStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode >= http.StatusInternalServerError
}

// queueRetryAfter parses a Retry-After header in seconds. Values that do
// not parse, or exceed the cap, are ignored (returning 0), keeping the
// regular polling schedule.
func queueRetryAfter(headers http.Header) time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(headers.Get("Retry-After")))
	if err != nil || seconds <= 0 {
		return 0
	}

	delay := time.Duration(seconds) * time.Second
	if delay > queueMaxRetryAfter {
		return queueMaxRetryAfter
	}

	return delay
}

// queueDecodeMediaBody decodes a raw provider media body into its bytes and
// parsed media type, validating the body's Content-Type against the required
// media prefix. It is the body of every queue-based media-output task
// codec's Decode; the PostSubmit exchange routes the downloaded output
// through that Decode too, so output media is validated the same way.
//
// CDNs commonly label any blob with a generic "application/octet-stream", so
// a body whose advertised type is unparseable — or parses outside the
// required prefix — is byte-sniffed before the request is rejected: a
// conclusive sniff matching the prefix wins, and the advertised type is
// rejected only when the bytes agree with it or cannot be identified at all.
func queueDecodeMediaBody(params DecodeParams, mediaPrefix string) ([]byte, string, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, mediaPrefix) {
		sniffed := http.DetectContentType(params.Body)
		if !strings.HasPrefix(sniffed, mediaPrefix) {
			return nil, "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "expected a " + mediaPrefix + "* Content-Type, got " + contentType,
				Err:     nil,
			}
		}

		return params.Body, sniffed, nil
	}

	return params.Body, mediaType, nil
}

// queueExchange binds the provider-specific steps of [queuePollOnce] to one
// task type: the task-specific decode and terminal-status classification.
type queueExchange[T any] struct {
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
		if queueIsContextError(err) {
			return nil, 0, err
		}

		return nil, 0, nil
	}

	if queueTransientStatus(resp.StatusCode) {
		override := queueRetryAfter(resp.Header)
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
