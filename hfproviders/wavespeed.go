package hfproviders

import (
	"context"
	"encoding/json"
	"net/http"
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

// wavespeedDefaultPollTiming returns the production polling schedule.
func wavespeedDefaultPollTiming() queueSchedule {
	return queueSchedule{
		initial: wavespeedInitialPollInterval,
		step:    wavespeedPollIntervalStep,
		max:     wavespeedMaxPollInterval,
	}
}

// wavespeedRouterPrefix is the router path segment WaveSpeed requests are
// proxied under.
const wavespeedRouterPrefix = "/wavespeed"

// WaveSpeedProvider serves text-to-video, text-to-image, and image-to-image
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
// output, surfacing the media bytes as the typed response. Adapter models —
// Hub mappings whose adapter is "lora" — automatically gain a "loras" payload
// entry pointing at their Hugging Face repository (see
// [wavespeedInjectAdapterLoras]). Provider-specific response metadata (task
// ID, timings) is exposed on the response's ProviderMetadata field.
// Extra provider-specific request parameters ride on the request's
// ProviderParams field.
//
// The zero value is safe to use: endpoint methods return a configuration
// [hferrors.SDKError] instead of dereferencing a nil resolver. Embedding all
// three mixins means the promoted field WaveSpeedProvider.Resolver is
// ambiguous — access each mixin's resolver explicitly if you ever need to
// (e.g. p.WaveSpeedTextToImageMixin.Resolver).
//
// As with the Hugging Face mixins, each WaveSpeed mixin can be embedded on its
// own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type WaveSpeedProvider struct {
	WaveSpeedTextToVideoMixin
	WaveSpeedTextToImageMixin
	WaveSpeedImageToImageMixin
}

// Compile-time assertions that WaveSpeedProvider satisfies each supported
// per-task provider interface.
var (
	_ ImageToImageProvider = (*WaveSpeedProvider)(nil)
	_ TextToImageProvider  = (*WaveSpeedProvider)(nil)
	_ TextToVideoProvider  = (*WaveSpeedProvider)(nil)
)

// NewWaveSpeedProvider creates a [WaveSpeedProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value WaveSpeedProvider is still safe to call; its
// endpoint methods return a configuration [hferrors.SDKError].
func NewWaveSpeedProvider(resolver ModelResolver) (WaveSpeedProvider, error) {
	if resolver == nil {
		return WaveSpeedProvider{}, errNilResolver(wavespeedProviderName)
	}

	return WaveSpeedProvider{
		WaveSpeedTextToVideoMixin:  WaveSpeedTextToVideoMixin{Resolver: resolver},
		WaveSpeedTextToImageMixin:  WaveSpeedTextToImageMixin{Resolver: resolver},
		WaveSpeedImageToImageMixin: WaveSpeedImageToImageMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the WaveSpeed provider for debug purposes.
func (p WaveSpeedProvider) Name() string {
	return wavespeedProviderName
}

// ProviderSuffix returns an empty string. WaveSpeed routes via the endpoint
// path, not a model suffix.
func (p WaveSpeedProvider) ProviderSuffix() string {
	return ""
}

// resolveWaveSpeedEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's provider model identifier and returns the router-proxied
// endpoint for the given task. The provider slug given to the resolver is
// WaveSpeed's Hub identifier — the key the Hub's mapping data uses — which is
// a provider fact, not something the SDK derives.
func resolveWaveSpeedEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
) (Endpoint, error) {
	if resolver == nil {
		return Endpoint{}, errNilResolver(wavespeedProviderName)
	}

	mapping, err := resolver.ResolveModel(params.Context, ResolveModelParams{
		Provider: wavespeedProviderName,
		Task:     task,
		Model:    params.Model,
	})
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    wavespeedProviderName + "/api/v3/" + mapping.ProviderModelID,
		Mapping: &mapping,
	}, nil
}

// WaveSpeedTextToVideoMixin provides the WaveSpeed endpoint and codec for
// text-to-video generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-video. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WaveSpeedTextToVideoMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToVideoEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WaveSpeedTextToVideoMixin) TextToVideoEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWaveSpeedEndpoint(m.Resolver, params, TaskTextToVideo)
}

// TextToVideoCodec returns the text-to-video codec, which encodes requests as
// WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated video bytes.
// The codec adapts the payload to adapter models with the mapping resolved at
// endpoint-construction time, falling back to the mixin's resolver when used
// standalone (see [wavespeedInjectAdapterLoras]).
func (m WaveSpeedTextToVideoMixin) TextToVideoCodec() TextToVideoCodec {
	return wavespeedTextToVideoCodec{wavespeedAsyncCodec{queueTaskCodec{
		defaults:    wavespeedDefaultPollTiming(),
		mediaPrefix: "video/",
		resolver:    m.Resolver,
	}}}
}

// WaveSpeedTextToImageMixin provides the WaveSpeed endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WaveSpeedTextToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WaveSpeedTextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWaveSpeedEndpoint(m.Resolver, params, TaskTextToImage)
}

// TextToImageCodec returns the text-to-image codec, which encodes requests as
// WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated image bytes.
// The codec adapts the payload to adapter models with the mapping resolved at
// endpoint-construction time, falling back to the mixin's resolver when used
// standalone (see [wavespeedInjectAdapterLoras]).
func (m WaveSpeedTextToImageMixin) TextToImageCodec() TextToImageCodec {
	return wavespeedTextToImageCodec{wavespeedAsyncCodec{queueTaskCodec{
		defaults:    wavespeedDefaultPollTiming(),
		mediaPrefix: "image/",
		resolver:    m.Resolver,
	}}}
}

// WaveSpeedImageToImageMixin provides the WaveSpeed endpoint and codec for
// image-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// image-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WaveSpeedImageToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// ImageToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WaveSpeedImageToImageMixin) ImageToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWaveSpeedEndpoint(m.Resolver, params, TaskImageToImage)
}

// ImageToImageCodec returns the image-to-image codec, which encodes requests
// as WaveSpeed's flat JSON body and completes WaveSpeed's asynchronous task
// exchange (poll and download) before returning the generated image bytes.
// The codec adapts the payload to adapter models with the mapping resolved at
// endpoint-construction time, falling back to the mixin's resolver when used
// standalone (see [wavespeedInjectAdapterLoras]).
func (m WaveSpeedImageToImageMixin) ImageToImageCodec() ImageToImageCodec {
	return wavespeedImageToImageCodec{wavespeedAsyncCodec{queueTaskCodec{
		defaults:    wavespeedDefaultPollTiming(),
		mediaPrefix: "image/",
		resolver:    m.Resolver,
	}}}
}

// wavespeedAsyncCodec is WaveSpeed's queue task codec: the shared
// [queueTaskCodec] base (resolver, polling knobs, media prefix, timing, and
// media decoding) plus WaveSpeed's media exchange steps. Codec structs
// embed it so their PostSubmit and Decode implementations share one exchange
// implementation; only the typed response construction differs per task.
type wavespeedAsyncCodec struct {
	queueTaskCodec
}

// postSubmitMedia runs the shared WaveSpeed exchange: decode the submit
// response, poll the result URL until a terminal status, and download the
// first output. It returns the raw media bytes, the download's response
// headers, and provider metadata extracted from the completed task; the
// caller validates the output and builds the typed response via its own
// Decode.
func (c wavespeedAsyncCodec) postSubmitMedia(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
) ([]byte, http.Header, map[string]any, error) {
	submitData, err := wavespeedDecodeTaskData(body)
	if err != nil {
		return nil, nil, nil, err
	}

	taskData, err := wavespeedAwaitResult(ctx, params, &submitData, c.timing())
	if err != nil {
		return nil, nil, nil, err
	}

	media, headers, err := queueDownloadOutput(ctx, params, taskData.Outputs[0])
	if err != nil {
		return nil, nil, nil, err
	}

	return media, headers, wavespeedMetadata(taskData), nil
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
		return wavespeedTaskData{}, queueDecodeTaskError("WaveSpeed task response", err)
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
// [queueDownloadOutput]) target WaveSpeed's CDN and are sent
// unauthenticated. Resolution is shared with the other queue providers (see
// [queuePollTarget]).
func wavespeedPollTarget(
	params PostSubmitParams,
	submitData *wavespeedTaskData,
) (request.Params, string, error) {
	if submitData.URLs.Get == "" {
		return request.Params{}, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "WaveSpeed task response has no result URL",
			Err:     nil,
		}
	}

	return queuePollTarget(
		params,
		submitData.URLs.Get,
		wavespeedRouterPrefix,
		"WaveSpeed result URL",
	)
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
	timing queueSchedule,
) (*wavespeedTaskData, error) {
	if submitData.Status == wavespeedStatusCompleted {
		return wavespeedCompletedWithOutputs(submitData)
	}
	if wavespeedIsFailedStatus(submitData.Status) {
		return nil, wavespeedTaskFailedError(submitData)
	}

	pollParams, pollPath, err := wavespeedPollTarget(params, submitData)
	if err != nil {
		return nil, err
	}

	poll := func(ctx context.Context) (*wavespeedTaskData, time.Duration, error) {
		return wavespeedPollOnce(ctx, params, pollParams, pollPath)
	}

	return queueAwait(ctx, timing, poll)
}

// wavespeedPollOnce issues one poll request and classifies the result. data is
// non-nil when the task reached a terminal status; retryAfter is a
// server-provided wait override for the next round; any error stops polling.
// The exchange shape is shared with the other queue providers (see
// [queuePollOnce]).
func wavespeedPollOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	resultPath string,
) (*wavespeedTaskData, time.Duration, error) {
	return queuePollOnce(ctx, params, pollParams, resultPath, queueExchange[wavespeedTaskData]{
		decode:   wavespeedDecodeTaskData,
		terminal: wavespeedTerminalStatus,
	})
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
		out, err = wavespeedCompletedWithOutputs(data)

		return out, true, err
	case wavespeedIsFailedStatus(data.Status):
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

// wavespeedCompletedWithOutputs validates a completed task's outputs and
// returns the task data when at least one output is available. A nil result
// with a nil error is never returned.
func wavespeedCompletedWithOutputs(data *wavespeedTaskData) (*wavespeedTaskData, error) {
	if len(data.Outputs) == 0 || data.Outputs[0] == "" {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "WaveSpeed task completed without outputs",
			Err:     nil,
		}
	}

	return data, nil
}

// wavespeedIsFailedStatus reports whether status is one of WaveSpeed's
// terminal failure statuses.
func wavespeedIsFailedStatus(status string) bool {
	switch status {
	case "failed", "cancelled", "timeout", "deleted":
		return true
	default:
		return false
	}
}

// wavespeedTaskFailedError builds the error surfaced for a terminally failed
// task, carrying the provider's error message and the task ID when present
// (see [queueTaskFailedError]).
func wavespeedTaskFailedError(data *wavespeedTaskData) error {
	message := "WaveSpeed task failed"
	if data.Error != "" {
		message += ": " + data.Error
	}
	if data.ID != "" {
		message += " (task id " + data.ID + ")"
	}

	return queueTaskFailedError(message)
}

// wavespeedNormalizeImageInput passes self-describing inputs through and
// wraps every other input as a JPEG data URI, matching the official Hugging
// Face clients' image-to-image payload preparation.
func wavespeedNormalizeImageInput(input string) string {
	if inputIsSelfDescribing(input) {
		return input
	}

	return "data:image/jpeg;base64," + input
}

// wavespeedInjectAdapterLoras mirrors the official Hugging Face client's LoRA
// handling for adapter models: when the model's Hub mapping declares a LoRA
// adapter (a fine-tune applied on top of a provider-hosted base model), the
// payload gains a "loras" entry pointing at the Hugging Face repository so
// the provider loads the adapter weights. Without it the request succeeds but
// runs the base model.
//
// The injected entry names the Hugging Face model ID with the default scale,
// exactly as huggingface.js does for WaveSpeed. Unlike huggingface.js, which
// overwrites any caller-supplied "loras", an entry already present in the
// payload wins, so callers can pass their own LoRA configuration through
// ProviderParams.
//
// mapping, when non-nil, is the resolution endpoint construction already
// performed (see [Endpoint.Mapping]) and is used directly, so no second
// lookup happens. Otherwise the resolver supplies the mapping, keeping
// standalone codec use working; a nil resolver then leaves the payload
// untouched. Lookup failures propagate: they are configuration errors, and in
// pipeline use an encode follows an endpoint resolution that already resolved
// the same mapping successfully.
func wavespeedInjectAdapterLoras(
	ctx context.Context,
	mapping *ModelMapping,
	resolver ModelResolver,
	task Task,
	hfModel string,
	payload map[string]any,
) error {
	if mapping == nil && resolver == nil {
		return nil
	}

	if mapping == nil {
		resolved, err := resolver.ResolveModel(ctx, ResolveModelParams{
			Provider: wavespeedProviderName,
			Task:     task,
			Model:    hfModel,
		})
		if err != nil {
			return err
		}
		mapping = &resolved
	}

	if mapping.Adapter != adapterLora {
		return nil
	}
	if _, ok := payload["loras"]; ok {
		return nil
	}

	payload["loras"] = []map[string]any{{"path": hfModel, "scale": 1}}

	return nil
}

// wavespeedEncodeHeaders returns the request headers shared by WaveSpeed
// encoders: a JSON body whose exact output subtype is provider-chosen, so
// requests advertise "Accept: */*".
func wavespeedEncodeHeaders() http.Header {
	return jsonRequestHeaders(mimeTypeAny)
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
	if err := mergeRequestParameters(payload, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}
	if err := wavespeedInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskTextToVideo, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
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
	media, headers, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	return queueFinalizeResponse(ctx, c.Decode, media, headers, metadata,
		func(resp *hftypes.TextToVideoResponse, meta map[string]any) {
			resp.ProviderMetadata = meta
		})
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
	if err := mergeRequestParameters(payload, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}
	if err := wavespeedInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskTextToImage, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
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
	media, headers, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	return queueFinalizeResponse(ctx, c.Decode, media, headers, metadata,
		func(resp *hftypes.TextToImageResponse, meta map[string]any) {
			resp.ProviderMetadata = meta
		})
}

// wavespeedImageToImageCodec is a [Codec] for image-to-image that encodes
// canonical requests into WaveSpeed's flat JSON body (the input image rides
// on both "image" and the required "images" array, the prompt comes from the
// parameters) and completes WaveSpeed's asynchronous task exchange (poll and
// download) before returning the generated image bytes.
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

	// WaveSpeed's body is flat: the source image rides on "image" and on
	// "images" (the array form models such as Bria Fibo require), and the
	// prompt (from the parameters) rides on "prompt". The dual-field shape
	// mirrors the official Hugging Face clients, which also send the input
	// as both a scalar and a one-element array; a provider that reads only
	// one of them ignores the other. ProviderParams may replace "images"
	// wholesale for multi-image models.
	image := wavespeedNormalizeImageInput(req.Input)
	payload := map[string]any{
		"image":  image,
		"images": []string{image},
	}
	if err := mergeRequestParameters(payload, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}
	if err := wavespeedInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskImageToImage, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
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
	media, headers, metadata, err := c.postSubmitMedia(ctx, params, body)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	return queueFinalizeResponse(ctx, c.Decode, media, headers, metadata,
		func(resp *hftypes.ImageToImageResponse, meta map[string]any) {
			resp.ProviderMetadata = meta
		})
}
