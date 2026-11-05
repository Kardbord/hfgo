package testutils

import (
	"context"
	"errors"
	"math/rand/v2"
	"net/http"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/stretchr/testify/require"
)

// statusServerOverloaded is the non-standard "server overloaded" status
// some inference providers return under load. net/http has no constant for
// it.
const statusServerOverloaded = 529

// transientStatus reports whether an HTTP status is a transient upstream
// failure the integration suite retries. The set is narrower than the
// provider polling loops' transient classification (see
// hfproviders.queueTransientStatus, which retries every 5xx): integration
// tests retry only statuses observed as idle-capacity or rate-limit blips
// from the live API, so deterministic server bugs (for example 501) surface
// on the first attempt instead of masking as retries. 429 is included
// because CI runs many packages in parallel against shared rate limits.
func transientStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout,
		statusServerOverloaded, http.StatusTooManyRequests:
		return true
	default:
		return false
	}
}

// transientSchedule holds the bounded-retry timing knobs for one
// RetryTransient call. Carrying the schedule per call instead of in
// package-level variables keeps the helper free of mutable shared state and
// lets tests pass a compressed schedule explicitly.
type transientSchedule struct {
	// maxRetries is the number of retries after the first attempt.
	maxRetries int

	// baseDelay is the wait before the first retry; each later retry waits
	// twice as long.
	baseDelay time.Duration

	// fallback429Delay is the wait for a 429 whose APIError reports no
	// server-provided Retry-After.
	fallback429Delay time.Duration

	// max429Delay caps a server-provided Retry-After honored by
	// [RetryTransient].
	max429Delay time.Duration
}

// Default schedule values. Their total bounded retry time stays near 14s
// of waiting plus request time.
const (
	// defaultMaxRetries is the number of retries after the first attempt.
	defaultMaxRetries = 3

	// defaultBaseDelay is the wait before the first retry; each later retry
	// waits twice as long.
	defaultBaseDelay = 2 * time.Second

	// default429FallbackDelay is the wait for a 429 whose APIError reports
	// no server-provided Retry-After.
	default429FallbackDelay = 10 * time.Second

	// default429MaxDelay caps a server-provided Retry-After honored by
	// [RetryTransient].
	default429MaxDelay = 30 * time.Second
)

// defaultTransientSchedule returns the schedule used when no option
// overrides it.
func defaultTransientSchedule() transientSchedule {
	return transientSchedule{
		maxRetries:       defaultMaxRetries,
		baseDelay:        defaultBaseDelay,
		fallback429Delay: default429FallbackDelay,
		max429Delay:      default429MaxDelay,
	}
}

// transientOption customizes the retry schedule of a RetryTransient call.
type transientOption func(*transientSchedule)

// withSchedule replaces the whole retry schedule with s.
func withSchedule(s transientSchedule) transientOption {
	return func(schedule *transientSchedule) {
		*schedule = s
	}
}

// jitterPct bounds the symmetric random jitter applied to each wait.
const jitterPct = 0.25

// IsTransient reports whether err is an [hferrors.APIError] with a status
// code the integration suite treats as a transient upstream failure worth
// retrying. 4xx statuses other than 429 — including 410 Gone, which the API
// uses for permanently removed or deprecated models — are never transient,
// nor are non-API errors such as [hferrors.SDKError] validation or
// configuration failures.
func IsTransient(err error) bool {
	var apiErr *hferrors.APIError
	if !errors.As(err, &apiErr) {
		return false
	}

	return transientStatus(apiErr.StatusCode)
}

// RetryTransient runs op and returns its value, retrying bounded transient
// upstream failures so a single 5xx/429 blip from the live API does not fail
// an integration test. op is called on each attempt; requests that are not
// idempotent may execute up to 1+schedule.maxRetries times.
//
// Only initiation calls are retried: an operation must not be wrapped if a
// failure occurs mid-stream, because a stream cannot be resumed and a
// retried initiation would restart from scratch with partial output
// discarded.
//
// Backoff doubles per retry starting at 2s, with symmetric jitter to keep
// parallel tests from retrying in lockstep. A 429 honors the server-provided
// Retry-After delay when the APIError carries one (capped at 30s), falling
// back to a fixed 10s otherwise. Each wait is aborted as soon as ctx is
// done, and the attempt loop stops immediately; the last error is reported
// via t and the test fails.
//
// Options may override the retry schedule; without options the default
// schedule is used.
func RetryTransient[T any](
	ctx context.Context,
	t *testing.T,
	op func() (T, error),
	opts ...transientOption,
) T {
	t.Helper()

	schedule := defaultTransientSchedule()
	for _, opt := range opts {
		opt(&schedule)
	}

	value, err := retryTransientErr(ctx, t, schedule, op)
	require.NoError(t, err, "operation did not succeed after retrying transient failures")

	return value
}

// retryTransientErr is [RetryTransient] without the final assertion so the
// bounded-retry behavior — attempts, backoff, 429 handling, context
// abort — is unit-testable on failing operations.
func retryTransientErr[T any](
	ctx context.Context,
	t *testing.T,
	sched transientSchedule,
	op func() (T, error),
) (T, error) {
	t.Helper()

	value, err := op()
	if err == nil {
		return value, nil
	}

	var zero T

	for attempt := 1; attempt <= sched.maxRetries; attempt++ {
		wait, waitErr := transientWait(err, attempt, sched)
		if waitErr != nil {
			break
		}
		if sleepCtx(ctx, wait) != nil {
			break
		}

		t.Logf("attempt %d/%d failed with %v; retrying in %s",
			attempt, sched.maxRetries+1, err, wait)

		value, err = op()
		if err == nil {
			return value, nil
		}
		if !IsTransient(err) {
			break
		}
	}

	return zero, err
}

// transientWait computes the wait before the next retry from the last
// attempt's error. A 429 with a server-provided Retry-After waits that
// delay; any other transient error follows the doubling backoff schedule.
// It returns an error when the error is not transient, in which case no
// further attempt is worthwhile.
func transientWait(err error, attempt int, sched transientSchedule) (time.Duration, error) {
	if !IsTransient(err) {
		return 0, errors.New("error is not transient")
	}

	var apiErr *hferrors.APIError
	// IsTransient already proved this conversion succeeds.
	_ = errors.As(err, &apiErr)

	if apiErr.StatusCode == http.StatusTooManyRequests {
		if serverDelay := apiErr.RetryAfter(); serverDelay > 0 {
			return cappedJitter(serverDelay, sched.max429Delay), nil
		}

		return jitter(sched.fallback429Delay), nil
	}

	return jitter(sched.baseDelay << (attempt - 1)), nil
}

// jitter applies symmetric random jitter of up to jitterPct around d.
func jitter(d time.Duration) time.Duration {
	factor := 1 + jitterPct - 2*jitterPct*rand.Float64() //nolint:gosec // jitter is not security-sensitive

	return time.Duration(float64(d) * factor)
}

// cappedJitter jitters d and caps the result at limit, preserving the
// limit's intent even when jitter happens to shrink the wait.
func cappedJitter(d, limit time.Duration) time.Duration {
	j := jitter(d)
	if j > limit {
		return limit
	}

	return j
}

// sleepCtx waits d, or until ctx is done, whichever comes first. It returns
// ctx.Err() when the context ended during the wait. A fresh timer per call
// avoids the time.After leak: time.After timers cannot be stopped, so they
// keep running until they fire even when the context exits first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)

	select {
	case <-ctx.Done():
		timer.Stop()

		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
