package hfproviders

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

const wavespeedProviderName = "wavespeed"

// WaveSpeed serving model, mirrored from the official Hugging Face clients:
// submitting a task returns a JSON prediction object whose result URL must be
// polled until the task reaches a terminal status, after which the first
// output URL is downloaded.
const (
	// wavespeedStatusCompleted marks a finished task whose outputs are ready.
	wavespeedStatusCompleted = "completed"
	// wavespeedStatusCreated marks a queued task.
	wavespeedStatusCreated = "created"
	// wavespeedStatusProcessing marks a running task.
	wavespeedStatusProcessing = "processing"
)

// Polling schedule bounds: WaveSpeed recommends waiting at least two seconds
// between polls and growing the interval toward five to ten seconds for
// long-running tasks.
const (
	// wavespeedInitialPollInterval is the wait before the first poll.
	wavespeedInitialPollInterval = 2 * time.Second
	// wavespeedPollIntervalStep is the per-poll wait increase.
	wavespeedPollIntervalStep = 1 * time.Second
	// wavespeedMaxPollInterval is the polling wait ceiling.
	wavespeedMaxPollInterval = 5 * time.Second
)

// wavespeedPollTiming bounds the polling loop's wait schedule: the first wait,
// the per-poll increase, and the wait ceiling.
type wavespeedPollTiming struct {
	initial time.Duration
	step    time.Duration
	max     time.Duration
}

// wavespeedDefaultPollTiming returns the production polling schedule.
func wavespeedDefaultPollTiming() wavespeedPollTiming {
	return wavespeedPollTiming{
		initial: wavespeedInitialPollInterval,
		step:    wavespeedPollIntervalStep,
		max:     wavespeedMaxPollInterval,
	}
}

// wavespeedRouterHost is the Hugging Face router host. When the caller's base
// URL targets it, provider-relative result paths resolve under the /wavespeed
// prefix, mirroring the official Hugging Face clients.
const wavespeedRouterHost = "router.huggingface.co"

// wavespeedMaxRetryAfter caps a server-provided Retry-After so a pathological
// header cannot stall the polling loop for an unbounded time.
const wavespeedMaxRetryAfter = 60 * time.Second

// WavespeedProvider serves text-to-video, text-to-image, and image-to-image
// tasks through the Hugging Face router using WaveSpeed's partner model
// identifiers. Callers configure the Hugging Face model ID they know; each
// embedded mixin translates it to WaveSpeed's identifier at
// endpoint-construction time via its [ModelResolver]. The Hugging Face router
// is a path-transparent proxy, so the endpoint path embeds the provider slug
// and WaveSpeed's own model ID.
//
// Wire format: requests are encoded as WaveSpeed's flat JSON body (the prompt
// and generation parameters at the top level, not nested under "parameters"),
// and responses follow WaveSpeed's asynchronous task API: the SDK submits,
// polls the result URL until the task completes, and downloads the first
// output, surfacing the media bytes as the typed response. Provider-specific
// response metadata (task ID, timings) is exposed on the response's
// ProviderMetadata field. Extra provider-specific request parameters ride on
// the request's ProviderParams field.
//
// The zero value is safe to use: endpoint methods return a configuration
// [hferrors.SDKError] instead of dereferencing a nil resolver. Embedding all
// three mixins means the promoted field WavespeedProvider.Resolver is
// ambiguous — access each mixin's resolver explicitly if you ever need to
// (e.g. p.WavespeedTextToImageMixin.Resolver).
//
// As with the Hugging Face mixins, each WaveSpeed mixin can be embedded on its
// own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type WavespeedProvider struct {
	WavespeedTextToVideoMixin
	WavespeedTextToImageMixin
	WavespeedImageToImageMixin
}

// Compile-time assertions that WavespeedProvider satisfies each supported
// per-task provider interface.
var (
	_ ImageToImageProvider = (*WavespeedProvider)(nil)
	_ TextToImageProvider  = (*WavespeedProvider)(nil)
	_ TextToVideoProvider  = (*WavespeedProvider)(nil)
)

// NewWavespeedProvider creates a [WavespeedProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value WavespeedProvider is still safe to call; its
// endpoint methods return a configuration [hferrors.SDKError].
func NewWavespeedProvider(resolver ModelResolver) (WavespeedProvider, error) {
	if resolver == nil {
		return WavespeedProvider{}, errNilResolver(wavespeedProviderName)
	}

	return WavespeedProvider{
		WavespeedTextToVideoMixin:  WavespeedTextToVideoMixin{Resolver: resolver},
		WavespeedTextToImageMixin:  WavespeedTextToImageMixin{Resolver: resolver},
		WavespeedImageToImageMixin: WavespeedImageToImageMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the WaveSpeed provider for debug purposes.
func (p WavespeedProvider) Name() string {
	return wavespeedProviderName
}

// ProviderSuffix returns an empty string. WaveSpeed routes via the endpoint
// path, not a model suffix.
func (p WavespeedProvider) ProviderSuffix() string {
	return ""
}

// resolveWavespeedEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's provider model identifier and returns the router-proxied
// endpoint for the given task. The provider slug given to the resolver is
// WaveSpeed's Hub identifier — the key the Hub's mapping data uses — which is
// a provider fact, not something the SDK derives.
func resolveWavespeedEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
) (Endpoint, error) {
	if resolver == nil {
		return Endpoint{}, errNilResolver(wavespeedProviderName)
	}

	providerModelID, err := resolver.ResolveModel(params.Context, ResolveModelParams{
		Provider: wavespeedProviderName,
		Task:     task,
		Model:    params.Model,
	})
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{Method: http.MethodPost, Path: "wavespeed/api/v3/" + providerModelID}, nil
}

// WavespeedTextToVideoMixin provides the WaveSpeed endpoint and codec for
// text-to-video generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-video. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedTextToVideoMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToVideoEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedTextToVideoMixin) TextToVideoEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskTextToVideo)
}

// TextToVideoCodec returns the text-to-video codec, which encodes requests as
// WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated video bytes.
func (WavespeedTextToVideoMixin) TextToVideoCodec() TextToVideoCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return wavespeedTextToVideoCodec{wavespeedAsyncCodec{
		mediaPrefix: "video/",
	}}
}

// WavespeedTextToImageMixin provides the WaveSpeed endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedTextToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedTextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskTextToImage)
}

// TextToImageCodec returns the text-to-image codec, which encodes requests as
// WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated image bytes.
func (WavespeedTextToImageMixin) TextToImageCodec() TextToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return wavespeedTextToImageCodec{wavespeedAsyncCodec{
		mediaPrefix: "image/",
	}}
}

// WavespeedImageToImageMixin provides the WaveSpeed endpoint and codec for
// image-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// image-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedImageToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// ImageToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedImageToImageMixin) ImageToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskImageToImage)
}

// ImageToImageCodec returns the image-to-image codec, which encodes requests
// as WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated image bytes.
func (WavespeedImageToImageMixin) ImageToImageCodec() ImageToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return wavespeedImageToImageCodec{wavespeedAsyncCodec{
		mediaPrefix: "image/",
	}}
}

// wavespeedAsyncCodec carries the polling schedule and expected output media
// prefix shared by every WaveSpeed task codec. Codec structs embed it so their
// PostSubmit and Decode implementations share one exchange implementation;
// only the typed response construction differs per task.
//
// The zero value is valid and uses the default polling schedule.
type wavespeedAsyncCodec struct {
	// mediaPrefix is the required output media-type prefix ("image/" or
	// "video/") enforced on the downloaded output.
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
func (c wavespeedAsyncCodec) timing() wavespeedPollTiming {
	timing := wavespeedDefaultPollTiming()
	if c.pollInterval > 0 {
		timing.initial = c.pollInterval
	}
	if c.pollStep > 0 {
		timing.step = c.pollStep
	}
	if c.maxPollInterval > 0 {
		timing.max = c.maxPollInterval
	}

	return timing
}

// decodeMediaBody decodes a raw provider media body into its bytes and parsed
// media type. It is the Decode fallback for direct media-body decoding: the
// task pipeline itself routes WaveSpeed responses through PostSubmit instead.
func (c wavespeedAsyncCodec) decodeMediaBody(params DecodeParams) ([]byte, string, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, c.mediaPrefix) {
		return nil, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected a " + c.mediaPrefix + "* Content-Type, got " + contentType,
			Err:     nil,
		}
	}

	return params.Body, mediaType, nil
}

// postSubmitMedia runs the shared WaveSpeed exchange: decode the submit
// response, poll the result URL until a terminal status, download the first
// output, and validate its media type. It returns the media bytes, parsed
// media type, and provider metadata extracted from the completed task.
func (c wavespeedAsyncCodec) postSubmitMedia(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
) ([]byte, string, map[string]any, error) {
	submitData, err := wavespeedDecodeTaskData(body)
	if err != nil {
		return nil, "", nil, err
	}

	taskData, err := wavespeedAwaitResult(ctx, params, &submitData, c.timing())
	if err != nil {
		return nil, "", nil, err
	}

	media, mediaType, err := wavespeedDownloadOutput(ctx, params, taskData.Outputs[0])
	if err != nil {
		return nil, "", nil, err
	}
	if !strings.HasPrefix(mediaType, c.mediaPrefix) {
		return nil, "", nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected a " + c.mediaPrefix + "* output, got " + mediaType,
			Err:     nil,
		}
	}

	return media, mediaType, wavespeedMetadata(taskData), nil
}

// wavespeedTaskURLs holds the task's result URL set.
type wavespeedTaskURLs struct {
	// Get is the provider-relative path (or absolute URL) polling reads the
	// task status from, e.g. "/api/v3/predictions/{id}/result".
	Get string `json:"get"`
}

// wavespeedTimings holds the task's timing breakdown.
type wavespeedTimings struct {
	Inference float64 `json:"inference"`
}

// wavespeedTaskData is the prediction object WaveSpeed returns on submit and
// on every poll.
type wavespeedTaskData struct {
	ID string `json:"id"`
	// Model is the provider model identifier the task ran on.
	Model           string            `json:"model"`
	Outputs         []string          `json:"outputs"`
	URLs            wavespeedTaskURLs `json:"urls"`
	HasNSFWContents []bool            `json:"has_nsfw_contents"`
	Status          string            `json:"status"`
	CreatedAt       string            `json:"created_at"`
	Error           string            `json:"error"`
	ExecutionTime   float64           `json:"executionTime"` //nolint:tagliatelle // WaveSpeed serves this field in camelCase.
	Timings         wavespeedTimings  `json:"timings"`
	// Code is WaveSpeed's response code; on a sync-mode wait timeout the
	// submit response carries 5004 while the task keeps processing. It does
	// not change polling behavior: the task is still pollable via URLs.Get.
	Code int `json:"code"`
}

// wavespeedTaskResponse is the envelope WaveSpeed wraps task data in.
type wavespeedTaskResponse struct {
	Code    int                `json:"code"`
	Message string             `json:"message"`
	Data    *wavespeedTaskData `json:"data"`
}

// wavespeedDecodeTaskData decodes a WaveSpeed task JSON body and returns its
// data object. Bodies that are not JSON — or JSON without a data object — are
// serialization errors.
func wavespeedDecodeTaskData(body []byte) (wavespeedTaskData, error) {
	var resp wavespeedTaskResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return wavespeedTaskData{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode WaveSpeed task response",
			Err:     err,
		}
	}
	if resp.Data == nil {
		return wavespeedTaskData{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "WaveSpeed task response missing data object",
			Err:     nil,
		}
	}

	return *resp.Data, nil
}

// wavespeedMetadata extracts the provider-specific response metadata exposed
// on the typed responses' ProviderMetadata field. It returns nil when the
// completed task carries nothing useful, so callers never observe an empty map.
func wavespeedMetadata(data *wavespeedTaskData) map[string]any {
	meta := make(map[string]any)
	if data.ID != "" {
		meta["id"] = data.ID
	}
	if data.Model != "" {
		meta["model"] = data.Model
	}
	if data.CreatedAt != "" {
		meta["created_at"] = data.CreatedAt
	}
	if data.ExecutionTime != 0 {
		meta["execution_time"] = data.ExecutionTime
	}
	if data.Timings.Inference != 0 {
		meta["timings"] = map[string]any{"inference": data.Timings.Inference}
	}
	if data.HasNSFWContents != nil {
		meta["has_nsfw_contents"] = data.HasNSFWContents
	}
	if len(meta) == 0 {
		return nil
	}

	return meta
}

// wavespeedPollTarget resolves the task's result URL and builds the request
// params for one poll. Polling always targets the provider host the submit
// went through, so the caller's bearer token applies. Output downloads (see
// wavespeedDownloadOutput) target WaveSpeed's CDN and are sent unauthenticated.
func wavespeedPollTarget(
	params PostSubmitParams,
	submitData *wavespeedTaskData,
) (request.Params, string, error) {
	resultURL, err := wavespeedResultURL(params.BaseURL, submitData.URLs.Get)
	if err != nil {
		return request.Params{}, "", err
	}

	pollParams := request.Params{
		HTTPClient: params.HTTPClient,
		BaseURL:    resultURL.base,
		UserAgent:  params.UserAgent,
		Token:      params.Token,
		Headers:    http.Header{"Accept": {mimeTypeJSON}},
	}
	pollPath := resultURL.path
	if resultURL.query != "" {
		pollPath += "?" + resultURL.query
	}

	return pollParams, pollPath, nil
}

// wavespeedAwaitResult polls the task's result URL until the task completes,
// then returns the completed task data. Callers must check Outputs before
// using the result. The context bounds the whole exchange, including polling:
// a task that outlives the context surfaces the context error, and the task
// itself keeps running server-side.
func wavespeedAwaitResult(
	ctx context.Context,
	params PostSubmitParams,
	submitData *wavespeedTaskData,
	timing wavespeedPollTiming,
) (*wavespeedTaskData, error) {
	if submitData.Status == wavespeedStatusCompleted {
		return completedWithOutputs(submitData)
	}
	if isWavespeedFailedStatus(submitData.Status) {
		return nil, wavespeedTaskFailedError(submitData)
	}

	pollParams, pollPath, err := wavespeedPollTarget(params, submitData)
	if err != nil {
		return nil, err
	}

	wait := timing.initial
	for {
		// nextWait grows the regular schedule toward the ceiling. A
		// server-provided Retry-After replaces the wait for the following
		// round, and the schedule then resumes growing from that value,
		// still capped by the ceiling.
		nextWait := min(wait+timing.step, timing.max)

		if err := wavespeedPollWait(ctx, wait); err != nil {
			return nil, err
		}

		data, override, err := wavespeedPollOnce(ctx, params, pollParams, pollPath)
		if err != nil {
			return nil, err
		}
		if data != nil {
			return data, nil
		}

		wait = nextWait
		if override > 0 {
			wait = override
		}
	}
}

// wavespeedPollWait waits d, or until ctx expires, whichever comes first. It
// returns ctx.Err() when the context ended during the wait.
func wavespeedPollWait(ctx context.Context, d time.Duration) error {
	// A fresh timer per call avoids the time.After leak: time.After timers
	// cannot be stopped, so they keep running until they fire even when the
	// context exits first. The drain after a failed Stop is safe because each
	// timer's channel is consumed exactly once.
	timer := time.NewTimer(d)

	select {
	case <-ctx.Done():
		if !timer.Stop() {
			<-timer.C
		}

		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// wavespeedPollOnce issues one poll request and classifies the result. data is
// non-nil when the task reached a terminal status; retryAfter is a
// server-provided wait override for the next round; any error stops polling.
func wavespeedPollOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	resultPath string,
) (*wavespeedTaskData, time.Duration, error) {
	//nolint:bodyclose // Drained and closed on transient and success branches;
	// non-transient 4xx/5xx errors transfer the body to the returned APIError,
	// which the caller must close per the APIError contract.
	resp, err := request.DoRaw(ctx, pollParams, http.MethodGet, resultPath, nil)
	if err != nil {
		// Transport-level failures (dropped connections, timeouts) are
		// transient: keep polling until the context expires, per the
		// provider's documented polling behavior. Context cancellation and
		// deadline expiry are not transient: they must surface immediately.
		if wavespeedIsContextError(err) {
			return nil, 0, err
		}

		return nil, 0, nil
	}

	if wavespeedTransientStatus(resp.StatusCode) {
		override := wavespeedRetryAfter(resp.Header)
		request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

		return nil, override, nil
	}
	// Non-transient error: transfer the open body to the returned APIError so
	// the caller can inspect it per the APIError contract.
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, 0, request.APIErrorFromResponse(resp, http.MethodGet)
	}

	body, readErr := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)
	if readErr != nil {
		return nil, 0, readErr
	}

	data, decodeErr := wavespeedDecodeTaskData(body)
	if decodeErr != nil {
		return nil, 0, decodeErr
	}

	out, done, terminalErr := wavespeedTerminalStatus(&data)
	if done {
		return out, 0, terminalErr
	}

	// Still running: keep polling on the regular schedule.
	return nil, 0, nil
}

// wavespeedIsContextError reports whether err is (or wraps) a context
// cancellation or deadline expiry. Such errors are not transient: they must
// stop polling immediately rather than be retried.
func wavespeedIsContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// wavespeedTerminalStatus classifies a decoded task's status. done reports
// whether the task reached a terminal status: true with a nil error means the
// task completed (out holds its data); true with an error means the task
// failed (err describes the failure); false means the task is still running
// (including unknown statuses), so polling should continue.
func wavespeedTerminalStatus(
	data *wavespeedTaskData,
) (out *wavespeedTaskData, done bool, err error) {
	switch {
	case data.Status == wavespeedStatusCompleted:
		out, err = completedWithOutputs(data)

		return out, true, err
	case isWavespeedFailedStatus(data.Status):
		return nil, true, wavespeedTaskFailedError(data)
	case data.Status == wavespeedStatusCreated, data.Status == wavespeedStatusProcessing:
		// Still running: keep polling on the regular schedule.
		return nil, false, nil
	default:
		// Unknown status: treat as in-progress and keep polling, so newly
		// introduced statuses do not abort long-running tasks.
		return nil, false, nil
	}
}

// wavespeedTransientStatus reports whether an HTTP status is a transient poll
// failure worth retrying (rate limiting and server errors).
func wavespeedTransientStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode >= http.StatusInternalServerError
}

// completedWithOutputs validates a completed task's outputs and returns the
// task data when at least one output is available. A nil result with a nil
// error is never returned.
func completedWithOutputs(data *wavespeedTaskData) (*wavespeedTaskData, error) {
	if len(data.Outputs) == 0 || data.Outputs[0] == "" {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "WaveSpeed task completed without outputs",
			Err:     nil,
		}
	}

	return data, nil
}

// isWavespeedFailedStatus reports whether status is one of WaveSpeed's
// terminal failure statuses.
func isWavespeedFailedStatus(status string) bool {
	switch status {
	case "failed", "cancelled", "timeout", "deleted":
		return true
	default:
		return false
	}
}

// wavespeedTaskFailedError builds the error surfaced for a terminally failed
// task, carrying the provider's error message and the task ID when present.
func wavespeedTaskFailedError(data *wavespeedTaskData) error {
	message := "WaveSpeed task failed"
	if data.Error != "" {
		message += ": " + data.Error
	}
	if data.ID != "" {
		message += " (task id " + data.ID + ")"
	}

	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: message,
		Err:     nil,
	}
}

// wavespeedResultURL resolves the task's poll URL against the caller's base
// URL. WaveSpeed returns the result path provider-relative (e.g.
// "/api/v3/predictions/{id}/result"); when the base URL is the Hugging Face
// router, the path resolves under the router's /wavespeed prefix, mirroring
// the official Hugging Face clients. An absolute URLs.Get is resolved from its
// path only, so the poll always targets the same host the submit did.
func wavespeedResultURL(baseURL, resultPath string) (wavespeedPolledURL, error) {
	if resultPath == "" {
		return wavespeedPolledURL{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "WaveSpeed task response has no result URL",
			Err:     nil,
		}
	}

	parsedBase, err := url.Parse(baseURL)
	if err != nil {
		return wavespeedPolledURL{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", baseURL),
			Err:     err,
		}
	}

	resultBase := baseURL
	if parsedBase.Host == wavespeedRouterHost {
		resultBase = strings.TrimSuffix(baseURL, "/") + "/wavespeed"
	}

	parsedPath, err := url.Parse(resultPath)
	if err != nil {
		return wavespeedPolledURL{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: fmt.Sprintf("invalid WaveSpeed result URL %q", resultPath),
			Err:     err,
		}
	}

	path := parsedPath.Path
	if path == "" || !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	return wavespeedPolledURL{
		base:  resultBase,
		path:  path,
		query: parsedPath.RawQuery,
	}, nil
}

// wavespeedPolledURL is the resolved poll request: a base URL to join against,
// a request path, and any query string the provider attached.
type wavespeedPolledURL struct {
	base  string
	path  string
	query string
}

// wavespeedRetryAfter parses a Retry-After header in seconds. Values that do
// not parse, or exceed the cap, are ignored (returning 0), keeping the
// regular polling schedule.
func wavespeedRetryAfter(headers http.Header) time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(headers.Get("Retry-After")))
	if err != nil || seconds <= 0 {
		return 0
	}

	delay := time.Duration(seconds) * time.Second
	if delay > wavespeedMaxRetryAfter {
		return wavespeedMaxRetryAfter
	}

	return delay
}

// wavespeedDownloadOutput downloads the task's first output URL. Output URLs
// are provider-CDN addresses (often signed), so the request goes through
// request.DoUnauthenticated, which structurally carries no caller
// credentials. The advertised Content-Type is preferred, falling back to
// byte-sniffing when the CDN omits it.
func wavespeedDownloadOutput(
	ctx context.Context,
	params PostSubmitParams,
	outputURL string,
) ([]byte, string, error) {
	//nolint:bodyclose // Drained and closed via defer below on success; error
	// responses transfer the body to the returned APIError, which the caller
	// must close per the APIError contract.
	resp, err := request.DoUnauthenticated(
		ctx,
		request.UnauthenticatedParams{
			HTTPClient: params.HTTPClient,
			UserAgent:  params.UserAgent,
		},
		http.MethodGet,
		outputURL,
		http.Header{"Accept": {"*/*"}},
		http.NoBody,
	)
	if err != nil {
		return nil, "", err
	}
	defer request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

	body, err := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	if err != nil {
		return nil, "", err
	}

	mediaType, ok := parseMediaType(resp.Header.Get("Content-Type"))
	if !ok {
		mediaType = http.DetectContentType(body)
	}

	return body, mediaType, nil
}

// wavespeedFlattenParameters marshals a typed parameters struct into a JSON
// object map so its non-nil fields can be merged into WaveSpeed's flat request
// body. Nil parameters yield an empty map.
func wavespeedFlattenParameters(parameters any) (map[string]any, error) {
	if parameters == nil {
		return map[string]any{}, nil
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to marshal inference parameters",
			Err:     err,
		}
	}

	flat := map[string]any{}
	if err := json.Unmarshal(raw, &flat); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to flatten inference parameters",
			Err:     err,
		}
	}

	return flat, nil
}

// wavespeedNormalizeImageInput passes data URIs and HTTP(S) URLs through and
// wraps every other input as a JPEG data URI, matching the official Hugging
// Face clients' image-to-image payload preparation.
func wavespeedNormalizeImageInput(input string) string {
	if input == "" ||
		strings.HasPrefix(input, "data:") ||
		strings.HasPrefix(input, "http://") ||
		strings.HasPrefix(input, "https://") {
		return input
	}

	return "data:image/jpeg;base64," + input
}

// wavespeedEncodeHeaders returns the request headers shared by WaveSpeed
// encoders: a JSON body whose exact output subtype is provider-chosen, so
// requests advertise "Accept: */*".
func wavespeedEncodeHeaders() http.Header {
	headers := http.Header{}
	headers.Set("Content-Type", mimeTypeJSON)
	headers.Set("Accept", "*/*")

	return headers
}

// wavespeedTextToVideoCodec is a [Codec] for text-to-video that encodes
// canonical requests into WaveSpeed's flat JSON body and completes WaveSpeed's
// asynchronous task exchange (poll and download) before returning the
// generated video bytes.
type wavespeedTextToVideoCodec struct {
	wavespeedAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToVideoRequest,
		hftypes.TextToVideoResponse,
	] = (*wavespeedTextToVideoCodec)(nil)
	_ TextToVideoCodec = (*wavespeedTextToVideoCodec)(nil)
)

func (c wavespeedTextToVideoCodec) Encode(
	params EncodeParams[hftypes.TextToVideoRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// WaveSpeed's body is flat: the prompt and every generation parameter
	// live at the top level.
	payload := map[string]any{"prompt": req.Input}
	if req.Parameters != nil {
		flat, err := wavespeedFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	maps.Copy(payload, req.ProviderParams)

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, wavespeedEncodeHeaders(), nil
}

func (c wavespeedTextToVideoCodec) Decode(
	params DecodeParams,
) (hftypes.TextToVideoResponse, error) {
	video, mediaType, err := c.decodeMediaBody(params)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	return hftypes.TextToVideoResponse{
		Video:            video,
		MediaType:        mediaType,
		ProviderMetadata: nil,
	}, nil
}

func (c wavespeedTextToVideoCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.TextToVideoResponse, error) {
	media, mediaType, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	return hftypes.TextToVideoResponse{
		Video:            media,
		MediaType:        mediaType,
		ProviderMetadata: metadata,
	}, nil
}

// wavespeedTextToImageCodec is a [Codec] for text-to-image that encodes
// canonical requests into WaveSpeed's flat JSON body and completes WaveSpeed's
// asynchronous task exchange (poll and download) before returning the
// generated image bytes.
type wavespeedTextToImageCodec struct {
	wavespeedAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToImageRequest,
		hftypes.TextToImageResponse,
	] = (*wavespeedTextToImageCodec)(nil)
	_ TextToImageCodec = (*wavespeedTextToImageCodec)(nil)
)

func (c wavespeedTextToImageCodec) Encode(
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// WaveSpeed's body is flat: the prompt and every generation parameter
	// live at the top level.
	payload := map[string]any{"prompt": req.Input}
	if req.Parameters != nil {
		flat, err := wavespeedFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	maps.Copy(payload, req.ProviderParams)

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, wavespeedEncodeHeaders(), nil
}

func (c wavespeedTextToImageCodec) Decode(
	params DecodeParams,
) (hftypes.TextToImageResponse, error) {
	image, mediaType, err := c.decodeMediaBody(params)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	return hftypes.TextToImageResponse{
		Image:            image,
		MediaType:        mediaType,
		ProviderMetadata: nil,
	}, nil
}

func (c wavespeedTextToImageCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.TextToImageResponse, error) {
	media, mediaType, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	return hftypes.TextToImageResponse{
		Image:            media,
		MediaType:        mediaType,
		ProviderMetadata: metadata,
	}, nil
}

// wavespeedImageToImageCodec is a [Codec] for image-to-image that encodes
// canonical requests into WaveSpeed's flat JSON body (input image as "image",
// prompt from the parameters) and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated image bytes.
type wavespeedImageToImageCodec struct {
	wavespeedAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.ImageToImageRequest,
		hftypes.ImageToImageResponse,
	] = (*wavespeedImageToImageCodec)(nil)
	_ ImageToImageCodec = (*wavespeedImageToImageCodec)(nil)
)

func (c wavespeedImageToImageCodec) Encode(
	params EncodeParams[hftypes.ImageToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// WaveSpeed's body is flat: the source image rides on "image" and the
	// prompt (from the parameters) rides on "prompt".
	payload := map[string]any{"image": wavespeedNormalizeImageInput(req.Input)}
	if req.Parameters != nil {
		flat, err := wavespeedFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	maps.Copy(payload, req.ProviderParams)

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, wavespeedEncodeHeaders(), nil
}

func (c wavespeedImageToImageCodec) Decode(
	params DecodeParams,
) (hftypes.ImageToImageResponse, error) {
	image, mediaType, err := c.decodeMediaBody(params)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	return hftypes.ImageToImageResponse{
		Image:            image,
		MediaType:        mediaType,
		ProviderMetadata: nil,
	}, nil
}

func (c wavespeedImageToImageCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.ImageToImageResponse, error) {
	media, mediaType, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	return hftypes.ImageToImageResponse{
		Image:            media,
		MediaType:        mediaType,
		ProviderMetadata: metadata,
	}, nil
}
