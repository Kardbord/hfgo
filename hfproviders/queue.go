package hfproviders

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/utils"
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

// queueSchedule bounds a polling loop's wait schedule: the wait before the
// first poll, the per-poll increase, and the wait ceiling.
type queueSchedule struct {
	initial time.Duration
	step    time.Duration
	max     time.Duration
}

// queueTiming merges optional per-codec overrides onto the default schedule:
// a non-positive override keeps the default value.
func queueTiming(defaults, overrides queueSchedule) queueSchedule {
	if overrides.initial > 0 {
		defaults.initial = overrides.initial
	}
	if overrides.step > 0 {
		defaults.step = overrides.step
	}
	if overrides.max > 0 {
		defaults.max = overrides.max
	}

	return defaults
}

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
	return utils.ParseRetryAfter(headers, queueMaxRetryAfter)
}

// queuePollWait waits d, or until ctx expires, whichever comes first. It
// returns ctx.Err() when the context ended during the wait.
func queuePollWait(ctx context.Context, d time.Duration) error {
	// A fresh timer per call avoids the time.After leak: time.After timers
	// cannot be stopped, so they keep running until they fire even when the
	// context exits first. Stop releases the timer; since Go 1.23 it reports
	// true whenever the value is undelivered, so no channel drain is needed.
	timer := time.NewTimer(d)

	select {
	case <-ctx.Done():
		timer.Stop()

		return ctx.Err()
	case <-timer.C:
		return nil
	}
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
			return nil, "", errUnexpectedMediaPrefix(mediaPrefix, contentType)
		}

		return params.Body, sniffed, nil
	}

	return params.Body, mediaType, nil
}

// queueTaskCodec carries the pieces shared by every queue-based task codec:
// the provider's default polling schedule, the resolver needed to adapt
// payloads to adapter models, and the expected output media prefix. The
// WaveSpeed, fal.ai, and Replicate codec types alias it so their Decode and
// PostSubmit implementations share one media-exchange and polling
// machinery; only the typed response construction differs per task.
//
// Every production construction goes through a provider mixin factory,
// which always sets defaults; the zero value is valid for standalone codec
// use, though a zero defaults leaves the polling waits at zero.
type queueTaskCodec struct {
	// defaults is the provider's production polling schedule. Any timing
	// knob below that stays unset keeps its default value.
	defaults queueSchedule

	// resolver maps the configured Hugging Face model ID to its provider
	// mapping. When the pipeline does not supply the mapping resolved at
	// endpoint-construction time (see [EncodeParams.Mapping]), Encode uses
	// the resolver to detect adapter models and inject the provider's LoRA
	// payload entry. Nil encodes without adapter handling unless a mapping
	// is supplied.
	resolver ModelResolver

	// mediaPrefix is the required output media-type prefix ("image/" or
	// "video/") enforced on the downloaded output. Unused by codecs whose
	// Decode consumes a JSON result body instead.
	mediaPrefix string

	// pollInterval overrides the initial polling wait (tests only). Zero
	// uses the default schedule.
	pollInterval time.Duration

	// pollStep overrides the per-poll wait increase (tests only). Zero uses
	// the default schedule.
	pollStep time.Duration

	// maxPollInterval overrides the polling wait ceiling (tests only). Zero
	// uses the default schedule.
	maxPollInterval time.Duration
}

// timing returns the codec's polling schedule, falling back to the defaults
// for any unset knob.
func (c queueTaskCodec) timing() queueSchedule {
	return queueTiming(c.defaults, queueSchedule{
		initial: c.pollInterval,
		step:    c.pollStep,
		max:     c.maxPollInterval,
	})
}

// decodeMediaBody decodes a raw provider media body into its bytes and
// parsed media type, validating the body's Content-Type against the codec's
// required media prefix (see [queueDecodeMediaBody]). Only the submit
// response bypasses it, routing through PostSubmit instead.
func (c queueTaskCodec) decodeMediaBody(params DecodeParams) ([]byte, string, error) {
	return queueDecodeMediaBody(params, c.mediaPrefix)
}

// queueTaskFailedError builds the error a queue provider's exchange
// surfaces for a terminally failed task. The provider reports the failure
// inside an otherwise valid response, so it surfaces as an SDKError of kind
// provider rather than an APIError: the HTTP exchange itself succeeded, and
// APIError stays reserved for 4xx/5xx responses. message is the assembled
// provider-labeled failure text.
func queueTaskFailedError(message string) error {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindProvider,
		Message: message,
		Err:     nil,
	}
}

// queueDecodeTaskError builds the serialization error surfaced when a queue
// provider's task response body cannot be decoded. label names the provider
// response object (for example "WaveSpeed task response").
func queueDecodeTaskError(label string, err error) *hferrors.SDKError {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: "failed to decode " + label,
		Err:     err,
	}
}

// queueTaskURL is a provider's task location for follow-up requests: its
// path and query, plus the host it named (empty for provider-relative
// paths) so callers can recognize a task URL that already points at the
// Hugging Face router.
type queueTaskURL struct {
	path  string
	query string
	host  string
}

// queueParseTaskURL parses a provider's task URL — a provider-relative path
// or an absolute URL — keeping only its path and query for follow-ups:
// requests always target the caller's base host, so a provider response can
// never redirect a token-bearing request to a foreign host. A bare path
// gains its leading slash, and a path with none normalizes to the root.
// label names the provider URL kind in the serialization error returned for
// an unparseable URL.
func queueParseTaskURL(rawURL, label string) (queueTaskURL, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return queueTaskURL{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: fmt.Sprintf("invalid %s %q", label, rawURL),
			Err:     err,
		}
	}

	path := parsed.Path
	if path == "" || !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	return queueTaskURL{path: path, query: parsed.RawQuery, host: parsed.Host}, nil
}

// queuePollBase resolves the base URL for a queue provider's credentialed
// follow-up requests. Polling always targets the host the submit went
// through, so follow-ups join the caller's base URL — stripped of its
// trailing slash — and when that base is the Hugging Face router they are
// proxied under routerPrefix. A taskURLHost already pointing at the router
// is the exception: its path carries the router prefix already, so the base
// stays unprefixed. routed reports the router verdict, so providers whose
// follow-up framing depends on it (fal.ai's queue-subdomain query) can
// adapt. An unparseable base is a configuration error: follow-ups cannot
// safely target it.
func queuePollBase(
	baseURL string,
	routerPrefix string,
	taskURLHost string,
) (base string, routed bool, err error) {
	parsedBase, err := url.Parse(baseURL)
	if err != nil {
		return "", false, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", baseURL),
			Err:     err,
		}
	}

	base = strings.TrimSuffix(baseURL, "/")
	routed = strings.EqualFold(parsedBase.Host, huggingfaceRouterHost)
	if !routed || strings.EqualFold(taskURLHost, huggingfaceRouterHost) {
		return base, routed, nil
	}

	return base + routerPrefix, true, nil
}

// queuePolledURL is a resolved queue follow-up request: a base URL to join
// against, a request path, and any query string the provider attached.
type queuePolledURL struct {
	base  string
	path  string
	query string
}

// queuePollURL resolves a provider's task URL against the caller's base URL
// for one credentialed follow-up (see [queueParseTaskURL] and
// [queuePollBase]).
func queuePollURL(
	baseURL string,
	rawURL string,
	routerPrefix string,
	label string,
) (queuePolledURL, error) {
	taskURL, err := queueParseTaskURL(rawURL, label)
	if err != nil {
		return queuePolledURL{}, err
	}

	base, _, err := queuePollBase(baseURL, routerPrefix, taskURL.host)
	if err != nil {
		return queuePolledURL{}, err
	}

	return queuePolledURL{base: base, path: taskURL.path, query: taskURL.query}, nil
}

// queuePollTarget resolves one credentialed queue follow-up GET: the
// request parameters carrying the caller's client, credentials, and
// User-Agent with a JSON Accept header, and the task URL's path with the
// provider's query joined back on (see [queuePollURL]).
func queuePollTarget(
	params PostSubmitParams,
	rawURL string,
	routerPrefix string,
	label string,
) (request.Params, string, error) {
	polled, err := queuePollURL(params.BaseURL, rawURL, routerPrefix, label)
	if err != nil {
		return request.Params{}, "", err
	}

	pollPath := polled.path
	if polled.query != "" {
		pollPath += "?" + polled.query
	}

	return request.Params{
		HTTPClient:           params.HTTPClient,
		BaseURL:              polled.base,
		UserAgent:            params.UserAgent,
		Token:                params.Token,
		Headers:              http.Header{headerAccept: {mimeTypeJSON}},
		MaxResponseBodyBytes: params.MaxResponseBodyBytes,
	}, pollPath, nil
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
	// non-transient 4xx/5xx errors have their bodies read and buffered into
	// the returned APIError, whose original response body APIErrorFromResponse
	// already closes.
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
	// Non-transient error: read the body into the returned APIError so the
	// caller can inspect it per the APIError contract.
	if resp.StatusCode >= http.StatusBadRequest {
		apiErr, apiErrErr := request.APIErrorFromResponse(
			resp, http.MethodGet, pollParams.MaxResponseBodyBytes,
		)
		if apiErrErr != nil {
			return nil, 0, apiErrErr
		}

		return nil, 0, apiErr
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

// queueAwait polls until pollOnce returns a terminal result, waiting
// schedule.initial before the first round and growing the wait by
// schedule.step toward schedule.max each round. A server-provided Retry-After
// override replaces the wait for the following round, after which the
// schedule resumes growing from that value, still capped by the ceiling.
func queueAwait[T any](
	ctx context.Context,
	schedule queueSchedule,
	pollOnce func(context.Context) (*T, time.Duration, error),
) (*T, error) {
	wait := schedule.initial
	for {
		// nextWait grows the regular schedule toward the ceiling. A
		// server-provided Retry-After replaces the wait for the following
		// round, and the schedule then resumes growing from that value,
		// still capped by the ceiling.
		nextWait := min(wait+schedule.step, schedule.max)

		if err := queuePollWait(ctx, wait); err != nil {
			return nil, err
		}

		out, override, err := pollOnce(ctx)
		if err != nil {
			return nil, err
		}
		if out != nil {
			return out, nil
		}

		wait = nextWait
		if override > 0 {
			wait = override
		}
	}
}

// queueFetchOnce issues one GET for a provider task resource (a result fetch
// or any follow-up read that needs the raw body without decode or
// terminal-status classification). It reports whether the attempt failed in a
// way worth retrying (a transport blip or a 429/5xx response) and carries a
// server-provided Retry-After wait for the next attempt. A non-transient
// error response stops the exchange as an APIError, and a read failure stops
// it as an SDKError. Context errors end the exchange immediately.
func queueFetchOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	path string,
) (body []byte, transient bool, override time.Duration, err error) {
	//nolint:bodyclose // Transient bodies are drained and closed below;
	// non-transient 4xx/5xx errors have their bodies read and buffered into
	// the returned APIError, whose original response body APIErrorFromResponse
	// already closes.
	resp, err := request.DoRaw(ctx, pollParams, http.MethodGet, path, nil)
	if err != nil {
		if queueIsContextError(err) {
			return nil, false, 0, err
		}

		return nil, true, 0, nil
	}

	if queueTransientStatus(resp.StatusCode) {
		override = queueRetryAfter(resp.Header)
		request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

		return nil, true, override, nil
	}
	if resp.StatusCode >= http.StatusBadRequest {
		apiErr, apiErrErr := request.APIErrorFromResponse(
			resp, http.MethodGet, pollParams.MaxResponseBodyBytes,
		)
		if apiErrErr != nil {
			return nil, false, 0, apiErrErr
		}

		return nil, false, 0, apiErr
	}

	body, readErr := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)
	if readErr != nil {
		return nil, false, 0, readErr
	}

	return body, false, 0, nil
}

// queueFetchRetry issues one fetch attempt per round, retrying transient
// failures on the schedule. The first attempt fires immediately; each
// subsequent attempt waits schedule.initial, grown by schedule.step toward
// schedule.max. A server-provided Retry-After override replaces the wait for
// the following round, after which the schedule resumes growing from that
// value, still capped by the ceiling. The task behind a result fetch is
// already complete — and billed — by the time this runs, so a rate limit or
// a server blip must not discard the generation.
func queueFetchRetry(
	ctx context.Context,
	schedule queueSchedule,
	fetchOnce func(context.Context) ([]byte, bool, time.Duration, error),
) ([]byte, error) {
	wait := schedule.initial
	for {
		// nextWait grows the regular schedule toward the ceiling. A
		// server-provided Retry-After replaces the wait for the following
		// round, and the schedule then resumes growing from that value,
		// still capped by the ceiling.
		nextWait := min(wait+schedule.step, schedule.max)

		body, transient, override, err := fetchOnce(ctx)
		if err != nil {
			return nil, err
		}
		if !transient {
			return body, nil
		}

		if err := queuePollWait(ctx, wait); err != nil {
			return nil, err
		}

		wait = nextWait
		if override > 0 {
			wait = override
		}
	}
}

// queueDownloadOutput downloads a provider-CDN output URL without caller
// credentials. Output URLs are provider-CDN addresses (often signed), so the
// request goes through request.DoUnauthenticated, which structurally carries
// no caller credentials. The returned headers always carry a parseable
// Content-Type: the advertised Content-Type is kept, falling back to
// byte-sniffing when the CDN omits or garbles it.
func queueDownloadOutput(
	ctx context.Context,
	params PostSubmitParams,
	outputURL string,
) ([]byte, http.Header, error) {
	//nolint:bodyclose // Drained and closed via defer below on success; error
	// responses have their bodies read and closed by APIErrorFromResponse.
	resp, err := request.DoUnauthenticated(
		ctx,
		request.UnauthenticatedParams{
			HTTPClient:           params.HTTPClient,
			UserAgent:            params.UserAgent,
			MaxResponseBodyBytes: params.MaxResponseBodyBytes,
		},
		http.MethodGet,
		outputURL,
		http.Header{headerAccept: {mimeTypeAny}},
		http.NoBody,
	)
	if err != nil {
		return nil, nil, err
	}
	defer request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

	body, err := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	if err != nil {
		return nil, nil, err
	}

	headers := resp.Header.Clone()
	if _, ok := parseMediaType(headers.Get("Content-Type")); !ok {
		headers.Set("Content-Type", http.DetectContentType(body))
	}

	return body, headers, nil
}

// queueFinalizeResponse is the shared tail of every queue media codec's
// PostSubmit: decode the downloaded output body through the codec's own
// Decode and attach the completed task's provider metadata. Response types
// are per-task hftypes DTOs with no shared interface, so each codec passes
// its own setter for the ProviderMetadata field. The helper does not constrain
// what decode returns, so attach must tolerate a zero or empty T.
func queueFinalizeResponse[T any](
	ctx context.Context,
	decode func(context.Context, DecodeParams) (T, error),
	media []byte,
	headers http.Header,
	metadata map[string]any,
	attach func(*T, map[string]any),
) (T, error) {
	resp, err := decode(ctx, DecodeParams{
		Body:    media,
		Event:   "", // unary responses have no SSE framing
		Headers: headers,
	})
	if err != nil {
		var zero T

		return zero, err
	}
	attach(&resp, metadata)

	return resp, nil
}
