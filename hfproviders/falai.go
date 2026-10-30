package hfproviders

import (
	"context"
	"encoding/base64"
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

const falaiProviderName = "fal-ai"

// falaiAdapterLora is the Hub mapping adapter value marking a model that
// ships LoRA weights applied on top of a provider-hosted base model.
const falaiAdapterLora = "lora"

// falaiRouterHost is the Hugging Face router host. When the caller's base
// URL targets it, provider-relative status and result paths resolve under
// the /fal-ai prefix, mirroring the official Hugging Face clients.
const falaiRouterHost = "router.huggingface.co"

// falaiRouterPrefix is the router path segment fal-ai requests are proxied
// under.
const falaiRouterPrefix = "/fal-ai"

// falaiQueueSubdomain is the query entry telling the router to proxy the
// request to fal's queue subdomain (queue.fal.run) instead of the
// synchronous fal.run endpoint. falaiQueueQuery is the same entry with the
// leading "?" for endpoint-path construction.
const (
	falaiQueueSubdomain = "_subdomain=queue"
	falaiQueueQuery     = "?" + falaiQueueSubdomain
)

// falaiLoraEndpoint is fal's generic SD-class LoRA endpoint. Models mapped to
// it must also name their base model in the payload, unlike flux-based
// LoRA endpoints.
const falaiLoraEndpoint = "fal-ai/lora"

// falaiLoraBaseModel is the base model fal requires for SD-class LoRAs
// served through falaiLoraEndpoint.
const falaiLoraBaseModel = "stabilityai/stable-diffusion-xl-base-1.0"

// falaiStatusCompleted is the only terminal queue status. A failed or
// cancelled request also reaches COMPLETED and surfaces its failure through
// the state's error fields, so the polling loop watches for the error field
// rather than a distinct failure status.
const falaiStatusCompleted = "COMPLETED"

// Polling schedule bounds: the loop starts at the official clients' 500 ms
// cadence and grows by 500 ms toward a 5 s ceiling, so long queue waits do
// not turn into unbounded request volume.
const (
	// falaiInitialPollInterval is the wait before the first poll.
	falaiInitialPollInterval = 500 * time.Millisecond
	// falaiPollIntervalStep is the per-poll wait increase.
	falaiPollIntervalStep = 500 * time.Millisecond
	// falaiMaxPollInterval is the polling wait ceiling.
	falaiMaxPollInterval = 5 * time.Second
)

// falaiMaxRetryAfter caps a server-provided Retry-After so a pathological
// header cannot stall the polling loop for an unbounded time.
const falaiMaxRetryAfter = 60 * time.Second

// Input data-URI media types for raw base64 inputs. fal's data-URL decoder
// keys off the declared label rather than the bytes, so these match the
// defaults the official Hugging Face clients send: image-to-image inputs as
// JPEG, segmentation inputs as PNG, and audio as MPEG regardless of its real
// container (the Hugging Face docs' fal example transcodes a FLAC file
// through the audio/mpeg label).
const (
	falaiImageInputMIME = "image/jpeg"
	falaiMaskInputMIME  = "image/png"
	falaiAudioInputMIME = "audio/mpeg"
)

// Downloaded-output media prefixes enforced by decodeMediaBody.
const (
	falaiImageMediaPrefix = "image/"
	falaiVideoMediaPrefix = "video/"
)

// Flat payload keys and canonical labels shared across fal.ai codecs.
const (
	falaiPromptKey = "prompt"
	falaiAcceptKey = "Accept"
	falaiMaskLabel = "mask"
)

// falaiPollTiming bounds the polling loop's wait schedule: the first wait,
// the per-poll increase, and the wait ceiling.
type falaiPollTiming struct {
	initial time.Duration
	step    time.Duration
	max     time.Duration
}

// falaiDefaultPollTiming returns the production polling schedule.
func falaiDefaultPollTiming() falaiPollTiming {
	return falaiPollTiming{
		initial: falaiInitialPollInterval,
		step:    falaiPollIntervalStep,
		max:     falaiMaxPollInterval,
	}
}

// FalAIProvider serves text-to-image, image-to-image, text-to-video,
// image-segmentation, and speech-recognition tasks through fal's generative
// media platform via the Hugging Face router. Callers configure the Hugging
// Face model ID they know; each embedded mixin translates it to fal's
// identifier at endpoint-construction time via its [ModelResolver]. The
// Hugging Face router is a path-transparent proxy, so the endpoint path
// embeds the provider slug and fal's own model ID — which itself begins with
// the "fal-ai/" namespace, making the routed path's doubled prefix
// intentional (e.g. "/fal-ai/fal-ai/flux/schnell").
//
// Wire format: every task runs through fal's queue API. The SDK submits the
// task, polls the task's status URL until it reaches the terminal COMPLETED
// status (500 ms initial wait growing toward 5 s; transient poll and
// result-fetch failures are retried and a server Retry-After is honored),
// fetches the result, and
// — for media tasks — downloads the first output URL, surfacing the media
// bytes or transcript as the typed response. A task that fails reports
// COMPLETED with an error payload; the SDK surfaces that as an
// [hferrors.SDKError] of kind provider. Provider-specific response metadata
// (request ID, queue position, inference time) is exposed on the response's
// ProviderMetadata field. Extra provider-specific request parameters ride
// on the request's ProviderParams field.
//
// Per-task payload adaptations mirror the official Hugging Face clients:
// canonical width/height parameters fold into fal's image_size object
// (text-to-image) and target_size renames to image_size (image-to-image);
// image and audio inputs are sent as data: URIs (raw base64 is labeled with
// the media types above, while data: and http(s) inputs pass through
// unchanged); image-segmentation forces sync_mode so the mask returns
// inline; and speech transcripts accept both whisper's text field and
// nemotron's output field, with chunk or segment timestamps mapped onto the
// canonical response. Adapter models — Hub mappings whose adapter is "lora"
// — automatically gain a "loras" payload entry pointing at the adapter
// weights on the Hub (see [falaiInjectAdapterLoras]).
//
// The zero value is safe to use: endpoint methods return a configuration
// [hferrors.SDKError] instead of dereferencing a nil resolver. Embedding all
// five mixins means the promoted field FalAIProvider.Resolver is ambiguous —
// access each mixin's resolver explicitly if you ever need to (e.g.
// p.FalAITextToImageMixin.Resolver).
//
// As with the Hugging Face mixins, each fal.ai mixin can be embedded on its
// own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type FalAIProvider struct {
	FalAITextToImageMixin
	FalAIImageToImageMixin
	FalAITextToVideoMixin
	FalAIImageSegmentationMixin
	FalAISpeechRecognitionMixin
}

// Compile-time assertions that FalAIProvider satisfies each supported
// per-task provider interface.
var (
	_ ImageToImageProvider      = (*FalAIProvider)(nil)
	_ TextToImageProvider       = (*FalAIProvider)(nil)
	_ TextToVideoProvider       = (*FalAIProvider)(nil)
	_ ImageSegmentationProvider = (*FalAIProvider)(nil)
	_ SpeechRecognitionProvider = (*FalAIProvider)(nil)
)

// NewFalAIProvider creates a [FalAIProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that
// fails at request time. A zero-value FalAIProvider is still safe to call;
// its endpoint methods return a configuration [hferrors.SDKError].
func NewFalAIProvider(resolver ModelResolver) (FalAIProvider, error) {
	if resolver == nil {
		return FalAIProvider{}, errNilResolver(falaiProviderName)
	}

	return FalAIProvider{
		FalAITextToImageMixin:       FalAITextToImageMixin{Resolver: resolver},
		FalAIImageToImageMixin:      FalAIImageToImageMixin{Resolver: resolver},
		FalAITextToVideoMixin:       FalAITextToVideoMixin{Resolver: resolver},
		FalAIImageSegmentationMixin: FalAIImageSegmentationMixin{Resolver: resolver},
		FalAISpeechRecognitionMixin: FalAISpeechRecognitionMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the fal.ai provider for debug purposes.
func (p FalAIProvider) Name() string {
	return falaiProviderName
}

// ProviderSuffix returns an empty string. fal.ai routes via the endpoint
// path, not a model suffix.
func (p FalAIProvider) ProviderSuffix() string {
	return ""
}

// resolveFalAIEndpoint resolves the configured Hugging Face model ID to
// fal's provider model identifier and returns the router-proxied queue
// endpoint for the given task. The provider slug given to the resolver is
// fal's Hub identifier — the key the Hub's mapping data uses — which is a
// provider fact, not something the SDK derives.
func resolveFalAIEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
) (Endpoint, error) {
	if resolver == nil {
		return Endpoint{}, errNilResolver(falaiProviderName)
	}

	mapping, err := resolver.ResolveModel(params.Context, ResolveModelParams{
		Provider: falaiProviderName,
		Task:     task,
		Model:    params.Model,
	})
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    falaiProviderName + "/" + mapping.ProviderModelID + falaiQueueQuery,
		Mapping: &mapping,
	}, nil
}

// FalAITextToImageMixin provides the fal.ai endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model
// ID to fal's provider model identifier at endpoint-construction time.
type FalAITextToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to fal's provider
	// model identifier. It must be non-nil when the mixin's endpoint method
	// is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// fal's identifier and returns the router-proxied queue path for it.
func (m FalAITextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveFalAIEndpoint(m.Resolver, params, TaskTextToImage)
}

// TextToImageCodec returns the text-to-image codec, which encodes requests
// as fal's flat JSON body and completes fal's queue exchange (submit, poll,
// fetch, download) before returning the generated image bytes. The codec
// adapts the payload to adapter models with the mapping resolved at
// endpoint-construction time, falling back to the mixin's resolver when used
// standalone (see [falaiInjectAdapterLoras]).
func (m FalAITextToImageMixin) TextToImageCodec() TextToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return falaiTextToImageCodec{falaiQueueCodec{
		mediaPrefix: falaiImageMediaPrefix,
		resolver:    m.Resolver,
	}}
}

// FalAIImageToImageMixin provides the fal.ai endpoint and codec for
// image-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// image-to-image. Its [ModelResolver] maps the configured Hugging Face model
// ID to fal's provider model identifier at endpoint-construction time.
type FalAIImageToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to fal's provider
	// model identifier. It must be non-nil when the mixin's endpoint method
	// is called.
	Resolver ModelResolver
}

// ImageToImageEndpoint resolves the configured Hugging Face model ID to
// fal's identifier and returns the router-proxied queue path for it.
func (m FalAIImageToImageMixin) ImageToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveFalAIEndpoint(m.Resolver, params, TaskImageToImage)
}

// ImageToImageCodec returns the image-to-image codec, which encodes requests
// as fal's flat JSON body (the input image rides on both "image_url" and the
// "image_urls" array, matching the official clients) and completes fal's
// queue exchange before returning the generated image bytes. The codec
// adapts the payload to adapter models with the mapping resolved at
// endpoint-construction time, falling back to the mixin's resolver when used
// standalone (see [falaiInjectAdapterLoras]).
func (m FalAIImageToImageMixin) ImageToImageCodec() ImageToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return falaiImageToImageCodec{falaiQueueCodec{
		mediaPrefix: falaiImageMediaPrefix,
		resolver:    m.Resolver,
	}}
}

// FalAITextToVideoMixin provides the fal.ai endpoint and codec for
// text-to-video generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-video. Its [ModelResolver] maps the configured Hugging Face model
// ID to fal's provider model identifier at endpoint-construction time.
type FalAITextToVideoMixin struct {
	// Resolver maps the configured Hugging Face model ID to fal's provider
	// model identifier. It must be non-nil when the mixin's endpoint method
	// is called.
	Resolver ModelResolver
}

// TextToVideoEndpoint resolves the configured Hugging Face model ID to
// fal's identifier and returns the router-proxied queue path for it.
func (m FalAITextToVideoMixin) TextToVideoEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveFalAIEndpoint(m.Resolver, params, TaskTextToVideo)
}

// TextToVideoCodec returns the text-to-video codec, which encodes requests
// as fal's flat JSON body and completes fal's queue exchange before
// returning the generated video bytes. The codec adapts the payload to
// adapter models with the mapping resolved at endpoint-construction time,
// falling back to the mixin's resolver when used standalone (see
// [falaiInjectAdapterLoras]).
func (m FalAITextToVideoMixin) TextToVideoCodec() TextToVideoCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return falaiTextToVideoCodec{falaiQueueCodec{
		mediaPrefix: falaiVideoMediaPrefix,
		resolver:    m.Resolver,
	}}
}

// FalAIImageSegmentationMixin provides the fal.ai endpoint and codec for
// image segmentation. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only image
// segmentation. Its [ModelResolver] maps the configured Hugging Face model
// ID to fal's provider model identifier at endpoint-construction time.
type FalAIImageSegmentationMixin struct {
	// Resolver maps the configured Hugging Face model ID to fal's provider
	// model identifier. It must be non-nil when the mixin's endpoint method
	// is called.
	Resolver ModelResolver
}

// ImageSegmentationEndpoint resolves the configured Hugging Face model ID to
// fal's identifier and returns the router-proxied queue path for it.
func (m FalAIImageSegmentationMixin) ImageSegmentationEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveFalAIEndpoint(m.Resolver, params, TaskImageSegmentation)
}

// ImageSegmentationCodec returns the image-segmentation codec, which encodes
// requests as fal's flat JSON body (forcing sync_mode so the mask returns
// inline) and completes fal's queue exchange before returning the mask
// segments.
func (m FalAIImageSegmentationMixin) ImageSegmentationCodec() ImageSegmentationCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return falaiImageSegmentationCodec{falaiQueueCodec{
		mediaPrefix: falaiImageMediaPrefix,
		resolver:    m.Resolver,
	}}
}

// FalAISpeechRecognitionMixin provides the fal.ai endpoint and codec for
// automatic speech recognition. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only speech
// recognition. Its [ModelResolver] maps the configured Hugging Face model ID
// to fal's provider model identifier at endpoint-construction time.
type FalAISpeechRecognitionMixin struct {
	// Resolver maps the configured Hugging Face model ID to fal's provider
	// model identifier. It must be non-nil when the mixin's endpoint method
	// is called.
	Resolver ModelResolver
}

// SpeechRecognitionEndpoint resolves the configured Hugging Face model ID to
// fal's identifier and returns the router-proxied queue path for it.
func (m FalAISpeechRecognitionMixin) SpeechRecognitionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveFalAIEndpoint(m.Resolver, params, TaskAutomaticSpeechRecognition)
}

// SpeechRecognitionCodec returns the speech-recognition codec, which encodes
// requests as fal's flat JSON body (the input audio rides on "audio_url" as
// a data: URI or URL) and completes fal's queue exchange before returning
// the transcript.
func (m FalAISpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs and media prefix are unused.
	return falaiSpeechRecognitionCodec{falaiQueueCodec{
		resolver: m.Resolver,
	}}
}

// falaiQueueCodec carries the pieces shared by every fal.ai task codec: the
// resolver needed to adapt payloads to adapter models, the polling schedule,
// and — for media-output tasks — the expected output media prefix. Codec
// structs embed it so their exchange implementations share one polling
// loop; only the result parsing and typed response construction differ per
// task.
//
// The zero value is valid and encodes payloads without adapter handling,
// using the default polling schedule.
type falaiQueueCodec struct {
	// resolver maps the configured Hugging Face model ID to its provider
	// mapping. When the pipeline does not supply the mapping resolved at
	// endpoint-construction time (see [EncodeParams.Mapping]), Encode uses
	// the resolver to detect adapter models and inject the provider's LoRA
	// payload entry (see [falaiInjectAdapterLoras]). Nil encodes without
	// adapter handling unless a mapping is supplied.
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
func (c falaiQueueCodec) timing() falaiPollTiming {
	timing := falaiDefaultPollTiming()
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

// decodeMediaBody validates a downloaded media body's Content-Type against
// the codec's required media prefix and returns its bytes and parsed media
// type. It is the body of every media-output task codec's Decode; the
// PostSubmit exchange routes the downloaded output through it too, so output
// media is validated the same way.
func (c falaiQueueCodec) decodeMediaBody(params DecodeParams) ([]byte, string, error) {
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

// postSubmitResult runs the shared fal.ai queue exchange: decode the submit
// response, poll the task's status URL until a terminal status, and fetch
// the result. It returns the completed task state (the metadata source) and
// the raw result JSON body; callers parse the task-specific output from it.
func (c falaiQueueCodec) postSubmitResult(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
) (*falaiTaskState, []byte, error) {
	submit, err := falaiDecodeTaskState(body)
	if err != nil {
		return nil, nil, err
	}

	pollParams, statusPath, resultPath, err := falaiQueueTargets(params, &submit)
	if err != nil {
		return nil, nil, err
	}

	state, err := c.awaitResult(ctx, params, pollParams, statusPath, &submit)
	if err != nil {
		return nil, nil, err
	}

	result, err := c.fetchResult(ctx, params, pollParams, resultPath)
	if err != nil {
		return nil, nil, err
	}

	return state, result, nil
}

// falaiMetrics holds the completed task's timing breakdown.
type falaiMetrics struct {
	InferenceTime *float64 `json:"inference_time"`
}

// falaiTaskState is the request state fal's queue returns on submit and on
// every status poll. The error fields are present only when the request
// failed; fal marks failed and cancelled requests COMPLETED like successful
// ones, so the error field — not the status — distinguishes them.
type falaiTaskState struct {
	RequestID   string `json:"request_id"`
	Status      string `json:"status"`
	ResponseURL string `json:"response_url"`
	// QueuePosition is present only while the request is IN_QUEUE.
	QueuePosition *int          `json:"queue_position"`
	Error         *string       `json:"error"`
	ErrorType     *string       `json:"error_type"`
	Metrics       *falaiMetrics `json:"metrics"`
}

// falaiDecodeTaskState decodes a fal.ai queue JSON body into its task state.
// Bodies that are not JSON — or JSON without a request ID and result URL —
// are serialization errors, because the exchange cannot continue without
// them.
func falaiDecodeTaskState(body []byte) (falaiTaskState, error) {
	var state falaiTaskState
	if err := json.Unmarshal(body, &state); err != nil {
		return falaiTaskState{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode fal.ai queue response",
			Err:     err,
		}
	}
	if state.RequestID == "" {
		return falaiTaskState{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai queue response missing request ID",
			Err:     nil,
		}
	}
	if state.ResponseURL == "" {
		return falaiTaskState{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai queue response missing response URL",
			Err:     nil,
		}
	}

	return state, nil
}

// falaiClassifyTask classifies a decoded task state. done reports whether
// the task reached the terminal status: true with a nil error means the task
// completed (the result is fetchable); true with an error means the task
// failed (err describes the failure); false means the task is still queued
// or running — including unknown statuses — so polling should continue.
func falaiClassifyTask(state *falaiTaskState) (done bool, err error) {
	if state.Status != falaiStatusCompleted {
		// IN_QUEUE, IN_PROGRESS, and any status fal introduces later: keep
		// polling so new intermediate states do not abort running tasks.
		return false, nil
	}
	if state.Error != nil && *state.Error != "" {
		return true, falaiTaskFailedError(state)
	}

	return true, nil
}

// falaiTaskFailedError builds the error surfaced for a terminally failed
// task, carrying the provider's error message and category plus the request
// ID when present. The provider reports the failure inside an otherwise
// valid response, so it surfaces as an SDKError of kind provider rather than
// an APIError: the HTTP exchange itself succeeded, and APIError stays
// reserved for 4xx/5xx responses.
func falaiTaskFailedError(state *falaiTaskState) error {
	message := "fal.ai task failed"
	if state.Error != nil && *state.Error != "" {
		message += ": " + *state.Error
	}
	if state.ErrorType != nil && *state.ErrorType != "" {
		message += " [" + *state.ErrorType + "]"
	}
	if state.RequestID != "" {
		message += " (request id " + state.RequestID + ")"
	}

	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindProvider,
		Message: message,
		Err:     nil,
	}
}

// falaiMetadata extracts the provider-specific response metadata exposed on
// the typed responses' ProviderMetadata field. It returns nil when the
// completed task carries nothing useful, so callers never observe an empty
// map.
func falaiMetadata(state *falaiTaskState) map[string]any {
	meta := make(map[string]any)
	if state.RequestID != "" {
		meta["request_id"] = state.RequestID
	}
	if state.QueuePosition != nil {
		meta["queue_position"] = *state.QueuePosition
	}
	if state.Metrics != nil && state.Metrics.InferenceTime != nil {
		meta["inference_time"] = *state.Metrics.InferenceTime
	}
	if len(meta) == 0 {
		return nil
	}

	return meta
}

// falaiQueueTargets resolves the credentialed status and result request
// targets. Polling always targets the host the submit went through: when the
// base URL is the Hugging Face router, the task's response-URL path resolves
// under the router's /fal-ai prefix and the queue-subdomain query rides
// along, mirroring the official Hugging Face clients. Only the response
// URL's path is used, so a provider response can never redirect a poll (and
// its bearer token) to a foreign host. Output downloads (see
// falaiDownloadOutput) target fal's CDN and are sent unauthenticated.
func falaiQueueTargets(
	params PostSubmitParams,
	submit *falaiTaskState,
) (pollParams request.Params, statusPath, resultPath string, err error) {
	base, routed, err := falaiPollBase(params.BaseURL)
	if err != nil {
		return request.Params{}, "", "", err
	}

	path, query, err := falaiResultPath(submit.ResponseURL)
	if err != nil {
		return request.Params{}, "", "", err
	}

	pollParams = request.Params{
		HTTPClient: params.HTTPClient,
		BaseURL:    base,
		UserAgent:  params.UserAgent,
		Token:      params.Token,
		Headers:    http.Header{falaiAcceptKey: {mimeTypeJSON}},
	}

	// Compose the follow-up query: the router's queue-subdomain directive
	// first when routed, then any query fal attached to the response URL.
	suffix := ""
	if routed {
		suffix = falaiQueueSubdomain
	}
	if query != "" {
		if suffix != "" {
			suffix += "&"
		}
		suffix += query
	}
	if suffix != "" {
		suffix = "?" + suffix
	}

	statusPath, resultPath = path+"/status"+suffix, path+suffix

	return pollParams, statusPath, resultPath, nil
}

// falaiPollBase resolves the base URL for credentialed follow-up requests
// and reports whether the submit went through the Hugging Face router (which
// requires the /fal-ai prefix and the queue-subdomain query on follow-ups).
func falaiPollBase(baseURL string) (base string, routed bool, err error) {
	parsedBase, err := url.Parse(baseURL)
	if err != nil {
		return "", false, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", baseURL),
			Err:     err,
		}
	}
	if strings.EqualFold(parsedBase.Host, falaiRouterHost) {
		return strings.TrimSuffix(baseURL, "/") + falaiRouterPrefix, true, nil
	}

	return baseURL, false, nil
}

// falaiResultPath extracts the request path and query from the task's
// response URL. fal serves response URLs on its own host (e.g.
// "https://queue.fal.run/fal-ai/flux/requests/{id}"); only the path and
// query are kept so follow-up requests target the host the submit went
// through, and any query fal attaches to the response URL is preserved
// (matching the WaveSpeed implementation).
func falaiResultPath(responseURL string) (path, query string, err error) {
	parsedPath, err := url.Parse(responseURL)
	if err != nil {
		return "", "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: fmt.Sprintf("invalid fal.ai response URL %q", responseURL),
			Err:     err,
		}
	}

	path = parsedPath.Path
	if path == "" || !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	return path, parsedPath.RawQuery, nil
}

// awaitResult polls the task's status URL until the task completes, then
// returns the completed task state. The context bounds the whole exchange,
// including polling: a task that outlives the context surfaces the context
// error, and the task itself keeps running server-side.
func (c falaiQueueCodec) awaitResult(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	statusPath string,
	submit *falaiTaskState,
) (*falaiTaskState, error) {
	if done, taskErr := falaiClassifyTask(submit); done {
		if taskErr != nil {
			return nil, taskErr
		}

		return submit, nil
	}

	timing := c.timing()
	wait := timing.initial
	for {
		// nextWait grows the regular schedule toward the ceiling. A
		// server-provided Retry-After replaces the wait for the following
		// round, and the schedule then resumes growing from that value,
		// still capped by the ceiling.
		nextWait := min(wait+timing.step, timing.max)

		if err := falaiPollWait(ctx, wait); err != nil {
			return nil, err
		}

		state, override, err := falaiPollOnce(ctx, params, pollParams, statusPath)
		if err != nil {
			return nil, err
		}
		if state != nil {
			return state, nil
		}

		wait = nextWait
		if override > 0 {
			wait = override
		}
	}
}

// falaiPollWait waits d, or until ctx expires, whichever comes first. It
// returns ctx.Err() when the context ended during the wait.
func falaiPollWait(ctx context.Context, d time.Duration) error {
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

// falaiPollOnce issues one status request and classifies the result. state
// is non-nil when the task reached a terminal status; retryAfter is a
// server-provided wait override for the next round; any error stops polling.
func falaiPollOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	statusPath string,
) (*falaiTaskState, time.Duration, error) {
	//nolint:bodyclose // Drained and closed on transient and success branches;
	// non-transient 4xx/5xx errors transfer the body to the returned APIError,
	// which the caller must close per the APIError contract.
	resp, err := request.DoRaw(ctx, pollParams, http.MethodGet, statusPath, nil)
	if err != nil {
		// Transport-level failures (dropped connections, timeouts) are
		// transient: keep polling until the context expires, per the
		// provider's documented polling behavior. Context cancellation and
		// deadline expiry are not transient: they must surface immediately.
		if falaiIsContextError(err) {
			return nil, 0, err
		}

		return nil, 0, nil
	}

	if falaiTransientStatus(resp.StatusCode) {
		override := falaiRetryAfter(resp.Header)
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

	state, decodeErr := falaiDecodeTaskState(body)
	if decodeErr != nil {
		return nil, 0, decodeErr
	}

	done, taskErr := falaiClassifyTask(&state)
	if done {
		return &state, 0, taskErr
	}

	// Still queued or running: keep polling on the regular schedule.
	return nil, 0, nil
}

// falaiIsContextError reports whether err is (or wraps) a context
// cancellation or deadline expiry. Such errors are not transient: they must
// stop polling immediately rather than be retried.
func falaiIsContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// falaiTransientStatus reports whether an HTTP status is a transient poll
// failure worth retrying (rate limiting and server errors).
func falaiTransientStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode >= http.StatusInternalServerError
}

// falaiRetryAfter parses a Retry-After header in seconds. Values that do
// not parse, or exceed the cap, are ignored (returning 0), keeping the
// regular polling schedule.
func falaiRetryAfter(headers http.Header) time.Duration {
	seconds, err := strconv.Atoi(strings.TrimSpace(headers.Get("Retry-After")))
	if err != nil || seconds <= 0 {
		return 0
	}

	delay := time.Duration(seconds) * time.Second
	if delay > falaiMaxRetryAfter {
		return falaiMaxRetryAfter
	}

	return delay
}

// fetchResult fetches the completed task's result JSON, retrying transient
// failures on the same schedule as the status poll. The task is already
// complete — and billed — by the time this runs, so a rate limit or a
// server blip must not discard the generation. Client errors surface
// immediately as an APIError: a failed task whose status carried no error
// is reported here, as fal returns the request error as a 4xx result
// response. Context errors end the exchange immediately. This goes beyond
// both official Hugging Face clients, whose result fetch is one-shot.
func (c falaiQueueCodec) fetchResult(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	resultPath string,
) ([]byte, error) {
	timing := c.timing()
	wait := timing.initial
	for {
		nextWait := min(wait+timing.step, timing.max)

		body, transient, override, err := falaiFetchResultOnce(ctx, params, pollParams, resultPath)
		if err != nil {
			return nil, err
		}
		if !transient {
			return body, nil
		}

		if err := falaiPollWait(ctx, wait); err != nil {
			return nil, err
		}

		wait = nextWait
		if override > 0 {
			wait = override
		}
	}
}

// falaiFetchResultOnce issues one result request. transient reports whether
// the attempt failed in a way worth retrying (a transport blip or a 429/5xx
// response); override is a server-provided wait for the next attempt. A
// non-transient error response — including a failed task's error — stops the
// exchange as an APIError, and a read failure stops it as an SDKError.
func falaiFetchResultOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	resultPath string,
) (body []byte, transient bool, override time.Duration, err error) {
	//nolint:bodyclose // Transient bodies are drained and closed below;
	// non-transient 4xx/5xx errors transfer the body to the returned
	// APIError, which the caller must close per the APIError contract.
	resp, err := request.DoRaw(ctx, pollParams, http.MethodGet, resultPath, nil)
	if err != nil {
		if falaiIsContextError(err) {
			return nil, false, 0, err
		}

		return nil, true, 0, nil
	}

	if falaiTransientStatus(resp.StatusCode) {
		override = falaiRetryAfter(resp.Header)
		request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)

		return nil, true, override, nil
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, false, 0, request.APIErrorFromResponse(resp, http.MethodGet)
	}

	body, readErr := request.ReadResponseBody(resp, params.MaxResponseBodyBytes)
	request.DrainAndCloseBody(resp.Body, params.MaxResponseBodyBytes)
	if readErr != nil {
		return nil, false, 0, readErr
	}

	return body, false, 0, nil
}

// falaiDownloadOutput downloads the task's first output URL. Output URLs are
// provider-CDN addresses (often signed), so the request goes through
// request.DoUnauthenticated, which structurally carries no caller
// credentials. The returned headers always carry a parseable Content-Type:
// the advertised Content-Type is kept, falling back to byte-sniffing when
// the CDN omits or garbles it.
func falaiDownloadOutput(
	ctx context.Context,
	params PostSubmitParams,
	outputURL string,
) ([]byte, http.Header, error) {
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
		http.Header{falaiAcceptKey: {"*/*"}},
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

// falaiMediaRef is a downloaded-output reference inside a fal result: an
// absolute CDN URL, or — for inline segmentation masks — a data: URI.
type falaiMediaRef struct {
	URL string `json:"url"`
}

// falaiImagesResult is the text-to-image and image-to-image result shape.
type falaiImagesResult struct {
	Images []falaiMediaRef `json:"images"`
}

// falaiVideoResult is the text-to-video result shape.
type falaiVideoResult struct {
	Video *falaiMediaRef `json:"video"`
}

// falaiSegmentationResult is the image-segmentation result shape. sync_mode
// responses inline the mask as a data: URI.
type falaiSegmentationResult struct {
	Image *falaiMediaRef `json:"image"`
}

// falaiImagesOutputURL extracts the first output URL from an images result
// body. An empty or missing image list is a serialization error.
func falaiImagesOutputURL(body []byte) (string, error) {
	var result falaiImagesResult
	if err := json.Unmarshal(body, &result); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode fal.ai image result",
			Err:     err,
		}
	}
	if len(result.Images) == 0 || result.Images[0].URL == "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai image result has no output URL",
			Err:     nil,
		}
	}

	return result.Images[0].URL, nil
}

// falaiVideoOutputURL extracts the output URL from a video result body. A
// missing video object or URL is a serialization error.
func falaiVideoOutputURL(body []byte) (string, error) {
	var result falaiVideoResult
	if err := json.Unmarshal(body, &result); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode fal.ai video result",
			Err:     err,
		}
	}
	if result.Video == nil || result.Video.URL == "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai video result has no output URL",
			Err:     nil,
		}
	}

	return result.Video.URL, nil
}

// falaiMaskRef extracts the mask reference from a segmentation result body.
func falaiMaskRef(body []byte) (string, error) {
	var result falaiSegmentationResult
	if err := json.Unmarshal(body, &result); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode fal.ai segmentation result",
			Err:     err,
		}
	}
	if result.Image == nil || result.Image.URL == "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai segmentation result has no mask URL",
			Err:     nil,
		}
	}

	return result.Image.URL, nil
}

// falaiInlineDataURL reports whether ref is a data URL and returns the
// payload after its first comma. fal's sync_mode masks are always
// base64-encoded data URLs, so the payload is already the canonical base64
// mask, exactly as the official Python client treats it.
func falaiInlineDataURL(ref string) (payload string, ok bool) {
	mediaType, payload, found := strings.Cut(ref, ",")
	if !found || !strings.HasPrefix(mediaType, "data:") {
		return "", false
	}

	return payload, true
}

// falaiASRChunk is a whisper-style timestamped chunk.
type falaiASRChunk struct {
	Text      string    `json:"text"`
	Timestamp []float64 `json:"timestamp"`
}

// falaiASRSegment is a nemotron-style transcript segment.
type falaiASRSegment struct {
	Text  string   `json:"text"`
	Start *float64 `json:"start"`
	End   *float64 `json:"end"`
}

// falaiASRResult is the speech-recognition result shape. The transcript
// rides on text (whisper) or output (nemotron); timestamps ride on chunks or
// segments.
type falaiASRResult struct {
	Text     *string           `json:"text"`
	Output   *string           `json:"output"`
	Chunks   []falaiASRChunk   `json:"chunks"`
	Segments []falaiASRSegment `json:"segments"`
}

// falaiDecodeASRResult decodes a speech-recognition result body into the
// canonical response. A result without any transcript is a serialization
// error; timestamps map onto canonical chunks, preferring whisper chunks and
// falling back to nemotron segments (open-ended segment bounds default to
// zero, as huggingface.js does).
func falaiDecodeASRResult(body []byte) (hftypes.SpeechRecognitionResponse, error) {
	var result falaiASRResult
	if err := json.Unmarshal(body, &result); err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode fal.ai speech recognition result",
			Err:     err,
		}
	}

	transcript, ok := falaiTranscript(&result)
	if !ok {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "fal.ai speech recognition result has no transcript",
			Err:     nil,
		}
	}

	return hftypes.SpeechRecognitionResponse{
		Text:             transcript,
		Chunks:           falaiASRChunks(&result),
		ProviderMetadata: nil,
	}, nil
}

// falaiTranscript extracts the transcript text, accepting whisper's text
// field and nemotron's output field. An empty string counts as a transcript
// only when the field is present, matching huggingface.js.
func falaiTranscript(result *falaiASRResult) (string, bool) {
	if result.Text != nil {
		return *result.Text, true
	}
	if result.Output != nil {
		return *result.Output, true
	}

	return "", false
}

// falaiASRChunks maps the result's timestamps onto canonical chunks. It
// returns nil when the result carries none.
func falaiASRChunks(result *falaiASRResult) []hftypes.SpeechRecognitionChunk {
	if len(result.Chunks) > 0 {
		chunks := make([]hftypes.SpeechRecognitionChunk, 0, len(result.Chunks))
		for _, chunk := range result.Chunks {
			chunks = append(chunks, hftypes.SpeechRecognitionChunk{
				Text:      chunk.Text,
				Timestamp: chunk.Timestamp,
			})
		}

		return chunks
	}

	if len(result.Segments) > 0 {
		chunks := make([]hftypes.SpeechRecognitionChunk, 0, len(result.Segments))
		for _, segment := range result.Segments {
			chunks = append(chunks, hftypes.SpeechRecognitionChunk{
				Text:      segment.Text,
				Timestamp: falaiSegmentTimestamp(segment.Start, segment.End),
			})
		}

		return chunks
	}

	return nil
}

// falaiSegmentTimestamp builds a two-element [start, end] timestamp,
// defaulting open bounds to zero.
func falaiSegmentTimestamp(start, end *float64) []float64 {
	var startSec, endSec float64
	if start != nil {
		startSec = *start
	}
	if end != nil {
		endSec = *end
	}

	return []float64{startSec, endSec}
}

// falaiFlattenParameters marshals a typed parameters struct into a JSON
// object map so its non-nil fields can be merged into fal's flat request
// body. Nil parameters yield an empty map.
func falaiFlattenParameters(parameters any) (map[string]any, error) {
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

// falaiNormalizeDataInput passes data URIs and HTTP(S) URLs through and
// wraps every other input as a data URI with the given media type, matching
// the official Hugging Face clients' payload preparation. fal's data-URL
// decoder keys off the declared label, so the label is a provider fact
// (see the input media-type constants above).
func falaiNormalizeDataInput(input, mediaType string) string {
	if input == "" ||
		strings.HasPrefix(input, "data:") ||
		strings.HasPrefix(input, "http://") ||
		strings.HasPrefix(input, "https://") {
		return input
	}

	return "data:" + mediaType + ";base64," + input
}

// falaiApplyImageSize folds a flat width/height parameter pair into fal's
// image_size object, as the official Python client does for text-to-image.
// A payload carrying only one of the two is left untouched, and a
// caller-supplied image_size (via ProviderParams, merged after this call)
// wins.
func falaiApplyImageSize(payload map[string]any) {
	width, hasWidth := payload["width"]
	height, hasHeight := payload["height"]
	if !hasWidth || !hasHeight {
		return
	}

	payload["image_size"] = map[string]any{"width": width, "height": height}
	delete(payload, "width")
	delete(payload, "height")
}

// falaiApplyTargetSize renames the canonical image-to-image target_size
// parameter to fal's image_size field, as the official Python client does.
// A payload without target_size is left untouched.
func falaiApplyTargetSize(payload map[string]any) {
	targetSize, ok := payload["target_size"]
	if !ok {
		return
	}

	payload["image_size"] = targetSize
	delete(payload, "target_size")
}

// falaiInjectAdapterLoras mirrors the official Hugging Face clients' LoRA
// handling for adapter models: when the model's Hub mapping declares a LoRA
// adapter (a fine-tune applied on top of a provider-hosted base model), the
// payload gains a "loras" entry pointing at the adapter weights on the Hub
// so the provider loads the adapter. Without it the request succeeds but
// runs the base model.
//
// The injected entry names the weights file's Hub resolve URL with the
// default scale, exactly as huggingface.js does. Models mapped to fal's
// generic SD-class LoRA endpoint additionally gain a "model_name" entry naming
// the base model, which fal requires for that endpoint but not for
// flux-based LoRA endpoints. Unlike huggingface.js, which overwrites
// caller-supplied "loras" and "model_name", the two entries are independent:
// a caller-supplied entry of either kind wins over its default, so passing a
// custom "loras" via ProviderParams still keeps the base-model default (and
// vice versa).
//
// mapping, when non-nil, is the resolution endpoint construction already
// performed (see [Endpoint.Mapping]) and is used directly, so no second
// lookup happens. Otherwise the resolver supplies the mapping, keeping
// standalone codec use working; a nil resolver then leaves the payload
// untouched. Lookup failures propagate: they are configuration errors, and
// in pipeline use an encode follows an endpoint resolution that already
// resolved the same mapping successfully.
func falaiInjectAdapterLoras(
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
			Provider: falaiProviderName,
			Task:     task,
			Model:    hfModel,
		})
		if err != nil {
			return err
		}
		mapping = &resolved
	}

	if mapping.Adapter != falaiAdapterLora || mapping.AdapterWeightsPath == "" {
		return nil
	}
	falaiApplyLoraBaseModel(task, mapping, payload)
	if _, ok := payload["loras"]; ok {
		return nil
	}

	payload["loras"] = []map[string]any{{
		"path":  DefaultHubBaseURL + "/" + hfModel + "/resolve/main/" + mapping.AdapterWeightsPath,
		"scale": 1,
	}}

	return nil
}

// falaiApplyLoraBaseModel names the base model for fal's generic SD-class
// LoRA endpoint, which — unlike dedicated LoRA endpoints whose base model
// is baked in — must be told which base to load. The default matches the
// constant both official Hugging Face clients send; a caller-supplied
// model_name wins.
func falaiApplyLoraBaseModel(task Task, mapping *ModelMapping, payload map[string]any) {
	if task != TaskTextToImage || mapping.ProviderModelID != falaiLoraEndpoint {
		return
	}
	if _, ok := payload["model_name"]; ok {
		return
	}

	payload["model_name"] = falaiLoraBaseModel
}

// falaiEncodeHeaders returns the request headers shared by fal.ai encoders:
// a JSON body whose submit response is always JSON.
func falaiEncodeHeaders() http.Header {
	headers := http.Header{}
	headers.Set("Content-Type", mimeTypeJSON)
	headers.Set("Accept", mimeTypeJSON)

	return headers
}

// falaiTextToImageCodec is a [Codec] for text-to-image that encodes
// canonical requests into fal's flat JSON body and completes fal's queue
// exchange (submit, poll, fetch, download) before returning the generated
// image bytes.
type falaiTextToImageCodec struct {
	falaiQueueCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToImageRequest,
		hftypes.TextToImageResponse,
	] = (*falaiTextToImageCodec)(nil)
	_ TextToImageCodec = (*falaiTextToImageCodec)(nil)
)

func (c falaiTextToImageCodec) Encode(
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// fal's body is flat: the prompt and every generation parameter live at
	// the top level. Canonical width/height fold into fal's image_size
	// object before ProviderParams merge, so callers can override image_size
	// directly.
	payload := map[string]any{falaiPromptKey: req.Input}
	if req.Parameters != nil {
		flat, err := falaiFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	falaiApplyImageSize(payload)
	maps.Copy(payload, req.ProviderParams)
	if err := falaiInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskTextToImage, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, falaiEncodeHeaders(), nil
}

func (c falaiTextToImageCodec) Decode(
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

func (c falaiTextToImageCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.TextToImageResponse, error) {
	state, result, err := c.postSubmitResult(ctx, params, body)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	outputURL, err := falaiImagesOutputURL(result)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	media, headers, err := falaiDownloadOutput(ctx, params, outputURL)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	resp, err := c.Decode(DecodeParams{
		Context: ctx,
		Body:    media,
		Event:   "", // unary responses have no SSE framing
		Headers: headers,
	})
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}
	resp.ProviderMetadata = falaiMetadata(state)

	return resp, nil
}

// falaiImageToImageCodec is a [Codec] for image-to-image that encodes
// canonical requests into fal's flat JSON body (the input image rides on
// both "image_url" and the "image_urls" array, matching the official
// Hugging Face clients' dual shape for endpoints that require one or the
// other) and completes fal's queue exchange before returning the generated
// image bytes.
type falaiImageToImageCodec struct {
	falaiQueueCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.ImageToImageRequest,
		hftypes.ImageToImageResponse,
	] = (*falaiImageToImageCodec)(nil)
	_ ImageToImageCodec = (*falaiImageToImageCodec)(nil)
)

func (c falaiImageToImageCodec) Encode(
	params EncodeParams[hftypes.ImageToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// fal's body is flat: the source image rides on "image_url" and on
	// "image_urls" (the array form endpoints such as FLUX.2-dev require),
	// and the prompt (from the parameters) rides on "prompt". The dual-field
	// shape mirrors the official Hugging Face clients; a provider that reads
	// only one of them ignores the other. ProviderParams may replace either
	// field wholesale.
	image := falaiNormalizeDataInput(req.Input, falaiImageInputMIME)
	payload := map[string]any{
		"image_url":  image,
		"image_urls": []string{image},
	}
	if req.Parameters != nil {
		flat, err := falaiFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	falaiApplyTargetSize(payload)
	maps.Copy(payload, req.ProviderParams)
	if err := falaiInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskImageToImage, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, falaiEncodeHeaders(), nil
}

func (c falaiImageToImageCodec) Decode(
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

func (c falaiImageToImageCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.ImageToImageResponse, error) {
	state, result, err := c.postSubmitResult(ctx, params, body)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	outputURL, err := falaiImagesOutputURL(result)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	media, headers, err := falaiDownloadOutput(ctx, params, outputURL)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	resp, err := c.Decode(DecodeParams{
		Context: ctx,
		Body:    media,
		Event:   "", // unary responses have no SSE framing
		Headers: headers,
	})
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}
	resp.ProviderMetadata = falaiMetadata(state)

	return resp, nil
}

// falaiTextToVideoCodec is a [Codec] for text-to-video that encodes
// canonical requests into fal's flat JSON body and completes fal's queue
// exchange before returning the generated video bytes.
type falaiTextToVideoCodec struct {
	falaiQueueCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToVideoRequest,
		hftypes.TextToVideoResponse,
	] = (*falaiTextToVideoCodec)(nil)
	_ TextToVideoCodec = (*falaiTextToVideoCodec)(nil)
)

func (c falaiTextToVideoCodec) Encode(
	params EncodeParams[hftypes.TextToVideoRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// fal's body is flat: the prompt and every generation parameter live at
	// the top level.
	payload := map[string]any{falaiPromptKey: req.Input}
	if req.Parameters != nil {
		flat, err := falaiFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	maps.Copy(payload, req.ProviderParams)
	if err := falaiInjectAdapterLoras(
		params.Context, params.Mapping, c.resolver, TaskTextToVideo, params.Model, payload,
	); err != nil {
		return nil, nil, err
	}

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, falaiEncodeHeaders(), nil
}

func (c falaiTextToVideoCodec) Decode(
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

func (c falaiTextToVideoCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.TextToVideoResponse, error) {
	state, result, err := c.postSubmitResult(ctx, params, body)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	outputURL, err := falaiVideoOutputURL(result)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	media, headers, err := falaiDownloadOutput(ctx, params, outputURL)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	resp, err := c.Decode(DecodeParams{
		Context: ctx,
		Body:    media,
		Event:   "", // unary responses have no SSE framing
		Headers: headers,
	})
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}
	resp.ProviderMetadata = falaiMetadata(state)

	return resp, nil
}

// falaiImageSegmentationCodec is a [Codec] for image segmentation that
// encodes canonical requests into fal's flat JSON body (forcing sync_mode
// so the mask returns inline) and completes fal's queue exchange before
// returning the mask segments. fal reports a single anonymous mask, so the
// canonical response carries one segment labeled "mask" with no score, as
// the official Python client produces.
type falaiImageSegmentationCodec struct {
	falaiQueueCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.ImageSegmentationRequest,
		[]hftypes.ImageSegmentation,
	] = (*falaiImageSegmentationCodec)(nil)
	_ ImageSegmentationCodec = (*falaiImageSegmentationCodec)(nil)
)

func (c falaiImageSegmentationCodec) Encode(
	params EncodeParams[hftypes.ImageSegmentationRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// fal's body is flat: the source image rides on "image_url" and the
	// segmentation parameters merge at the top level. sync_mode is forced
	// last — after ProviderParams — so the mask returns inline exactly as
	// the official Hugging Face clients request it.
	payload := map[string]any{
		"image_url": falaiNormalizeDataInput(req.Input, falaiMaskInputMIME),
	}
	if req.Parameters != nil {
		flat, err := falaiFlattenParameters(req.Parameters)
		if err != nil {
			return nil, nil, err
		}
		maps.Copy(payload, flat)
	}
	maps.Copy(payload, req.ProviderParams)
	payload["sync_mode"] = true

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, falaiEncodeHeaders(), nil
}

func (c falaiImageSegmentationCodec) Decode(
	params DecodeParams,
) ([]hftypes.ImageSegmentation, error) {
	mask, _, err := c.decodeMediaBody(params)
	if err != nil {
		return nil, err
	}

	return []hftypes.ImageSegmentation{{
		Label:            falaiMaskLabel,
		Mask:             base64.StdEncoding.EncodeToString(mask),
		Score:            nil,
		ProviderMetadata: nil,
	}}, nil
}

func (c falaiImageSegmentationCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) ([]hftypes.ImageSegmentation, error) {
	state, result, err := c.postSubmitResult(ctx, params, body)
	if err != nil {
		return nil, err
	}

	maskRef, err := falaiMaskRef(result)
	if err != nil {
		return nil, err
	}

	metadata := falaiMetadata(state)

	// sync_mode responses inline the mask as a base64 data URL: its payload
	// is the canonical mask, so no download happens.
	if payload, ok := falaiInlineDataURL(maskRef); ok {
		return []hftypes.ImageSegmentation{{
			Label:            falaiMaskLabel,
			Mask:             payload,
			Score:            nil,
			ProviderMetadata: metadata,
		}}, nil
	}

	media, headers, err := falaiDownloadOutput(ctx, params, maskRef)
	if err != nil {
		return nil, err
	}

	segments, err := c.Decode(DecodeParams{
		Context: ctx,
		Body:    media,
		Event:   "", // unary responses have no SSE framing
		Headers: headers,
	})
	if err != nil {
		return nil, err
	}
	segments[0].ProviderMetadata = metadata

	return segments, nil
}

// falaiSpeechRecognitionCodec is a [Codec] for speech recognition that
// encodes canonical requests into fal's flat JSON body (the input audio
// rides on "audio_url" as a data: URI or URL) and completes fal's queue
// exchange before returning the transcript.
type falaiSpeechRecognitionCodec struct {
	falaiQueueCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.SpeechRecognitionRequest,
		hftypes.SpeechRecognitionResponse,
	] = (*falaiSpeechRecognitionCodec)(nil)
	_ SpeechRecognitionCodec = (*falaiSpeechRecognitionCodec)(nil)
)

func (c falaiSpeechRecognitionCodec) Encode(
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	// fal's body is flat: the source audio rides on "audio_url" and the
	// recognition parameters (e.g. return_timestamps) merge at the top
	// level.
	payload := map[string]any{
		"audio_url": falaiNormalizeDataInput(req.Input, falaiAudioInputMIME),
	}
	if req.Parameters != nil {
		flat, err := falaiFlattenParameters(req.Parameters)
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

	return body, falaiEncodeHeaders(), nil
}

func (c falaiSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return falaiDecodeASRResult(params.Body)
}

func (c falaiSpeechRecognitionCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.SpeechRecognitionResponse, error) {
	state, result, err := c.postSubmitResult(ctx, params, body)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	resp, err := c.Decode(DecodeParams{
		Context: ctx,
		Body:    result,
		Event:   "", // unary responses have no SSE framing
		Headers: http.Header{"Content-Type": {mimeTypeJSON}},
	})
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}
	resp.ProviderMetadata = falaiMetadata(state)

	return resp, nil
}
