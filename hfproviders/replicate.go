package hfproviders

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

const replicateProviderName = "replicate"

// replicateRouterPrefix is the router path segment replicate requests are
// proxied under.
const replicateRouterPrefix = "/replicate"

// Replicate prediction statuses. A prediction is terminal once it reaches
// succeeded, failed, or canceled; every other status — including the ones
// Replicate documents today (starting, processing) and any it introduces
// later — keeps the polling loop running.
const (
	// replicateStatusSucceeded marks a finished prediction whose output is
	// ready.
	replicateStatusSucceeded = "succeeded"
	// replicateStatusFailed marks a prediction that errored out.
	replicateStatusFailed = "failed"
	// replicateStatusCanceled marks a prediction that was canceled.
	replicateStatusCanceled = "canceled"
)

// Polling schedule bounds: the loop starts at two seconds and grows by one
// second toward a five-second ceiling, so long generation waits do not turn
// into unbounded request volume. Replicate predictions — video generation
// and cold-started models in particular — routinely run for minutes, so the
// loop is bounded by the caller's context rather than a poll count.
const (
	// replicateInitialPollInterval is the wait before the first poll.
	replicateInitialPollInterval = 2 * time.Second
	// replicatePollIntervalStep is the per-poll wait increase.
	replicatePollIntervalStep = 1 * time.Second
	// replicateMaxPollInterval is the polling wait ceiling.
	replicateMaxPollInterval = 5 * time.Second
)

// Input data-URI media-type fallbacks for raw base64 inputs. They match the
// defaults the official Hugging Face clients label undecodable inputs with;
// whenever byte sniffing identifies the container, the sniffed type wins.
const (
	replicateAudioInputMIME = "audio/wav"
	replicateImageInputMIME = "image/jpeg"
)

// Downloaded-output media prefixes enforced by decodeMediaBody.
const (
	replicateImageMediaPrefix = "image/"
	replicateVideoMediaPrefix = "video/"
)

// Payload keys and canonical labels shared across replicate codecs.
const (
	replicateInputKey       = "input"
	replicateVersionKey     = "version"
	replicatePromptKey      = "prompt"
	replicateAudioKey       = "audio"
	replicateLoraWeightsKey = "lora_weights"
	replicatePreferKey      = "Prefer"
	replicatePreferWait     = "wait"
)

// ASR output object keys, matching the fields multilingual Replicate models
// return alongside a plain transcript.
const (
	replicateTranscriptionKey = "transcription"
	replicateTranslationKey   = "translation"
	replicateTxtFileKey       = "txt_file"
)

// replicateDefaultPollTiming returns the production polling schedule.
func replicateDefaultPollTiming() queueSchedule {
	return queueSchedule{
		initial: replicateInitialPollInterval,
		step:    replicatePollIntervalStep,
		max:     replicateMaxPollInterval,
	}
}

// replicateErrSucceededWithoutOutput is the serialization error surfaced
// when a succeeded prediction carries no usable output. The terminal-status
// classifier and the output extractor share it: reaching either with no
// output means the same malformed completion.
const replicateErrSucceededWithoutOutput = "Replicate prediction succeeded without output"

// ReplicateProvider serves text-to-video, text-to-image, image-to-image,
// and automatic-speech-recognition tasks through Replicate via the Hugging
// Face router. Callers configure the Hugging Face model ID they know; each
// embedded mixin translates it to Replicate's identifier at
// endpoint-construction time via its [ModelResolver]. The Hugging Face
// router is a path-transparent proxy, so the endpoint path embeds the
// provider slug and Replicate's own model-scoped prediction route.
//
// Wire format: every task runs through Replicate's prediction API. The SDK
// submits the prediction with the "Prefer: wait" header, which asks
// Replicate to hold the request open until the prediction finishes — the
// provider caps that server-side wait (about a minute; cold-starting models
// and video generation routinely exceed it). When the wait window expires
// before the prediction finishes, the SDK polls the prediction's URL until
// it reaches a terminal status, then downloads the first output URL —
// surfacing the media bytes or transcript as the typed response. This
// client-side polling goes beyond the official Hugging Face clients, which
// give up when the wait window expires; it is what makes cold-start and
// long-running models usable. A prediction that fails surfaces as an
// [hferrors.SDKError] of kind provider. Provider-specific response metadata
// (prediction ID, model, status, creation time, timing breakdown, logs) is
// exposed on the response's ProviderMetadata field. Extra provider-specific
// request parameters ride on the request's ProviderParams field.
//
// Per-task payload adaptations mirror the official Hugging Face clients:
// generation prompts ride on "prompt" inside the prediction's "input"
// object; image-to-image inputs are sent under all four keys different
// Replicate models expect ("image", "images", "input_image",
// "input_images"); speech-recognition audio and image-to-image inputs ride
// as data: URIs (raw base64 is labeled with a sniffed media type, falling
// back to the official clients' defaults); adapter models — Hub mappings
// whose adapter is "lora" — automatically gain a "lora_weights" entry
// pointing at their Hugging Face repository on text-to-image and
// image-to-image, matching the official Hugging Face clients; and
// version-pinned mappings (provider identifiers containing ":") submit
// through the generic predictions endpoint with the version hash in the
// payload.
//
// The zero value is safe to use: endpoint methods return a configuration
// [hferrors.SDKError] instead of dereferencing a nil resolver. Embedding all
// four mixins means the promoted field ReplicateProvider.Resolver is
// ambiguous — access each mixin's resolver explicitly if you ever need to
// (e.g. p.ReplicateTextToImageMixin.Resolver).
//
// As with the Hugging Face mixins, each Replicate mixin can be embedded on
// its own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type ReplicateProvider struct {
	ReplicateTextToImageMixin
	ReplicateImageToImageMixin
	ReplicateTextToVideoMixin
	ReplicateSpeechRecognitionMixin
}

// Compile-time assertions that ReplicateProvider satisfies each supported
// per-task provider interface.
var (
	_ ImageToImageProvider      = (*ReplicateProvider)(nil)
	_ SpeechRecognitionProvider = (*ReplicateProvider)(nil)
	_ TextToImageProvider       = (*ReplicateProvider)(nil)
	_ TextToVideoProvider       = (*ReplicateProvider)(nil)
)

// NewReplicateProvider creates a [ReplicateProvider] backed by the given
// model resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that
// fails at request time. A zero-value ReplicateProvider is still safe to
// call; its endpoint methods return a configuration [hferrors.SDKError].
func NewReplicateProvider(resolver ModelResolver) (ReplicateProvider, error) {
	if resolver == nil {
		return ReplicateProvider{}, errNilResolver(replicateProviderName)
	}

	return ReplicateProvider{
		ReplicateTextToImageMixin: ReplicateTextToImageMixin{
			Resolver: resolver,
		},
		ReplicateImageToImageMixin: ReplicateImageToImageMixin{
			Resolver: resolver,
		},
		ReplicateTextToVideoMixin: ReplicateTextToVideoMixin{
			Resolver: resolver,
		},
		ReplicateSpeechRecognitionMixin: ReplicateSpeechRecognitionMixin{
			Resolver: resolver,
		},
	}, nil
}

// Name returns the name of the Replicate provider for debug purposes.
func (p ReplicateProvider) Name() string {
	return replicateProviderName
}

// ProviderSuffix returns an empty string. Replicate routes via the endpoint
// path, not a model suffix.
func (p ReplicateProvider) ProviderSuffix() string {
	return ""
}

// resolveReplicateEndpoint resolves the configured Hugging Face model ID to
// Replicate's provider model identifier and returns the router-proxied
// prediction endpoint for the given task. The provider slug given to the
// resolver is Replicate's Hub identifier — the key the Hub's mapping data
// uses — which is a provider fact, not something the SDK derives.
func resolveReplicateEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
) (Endpoint, error) {
	if resolver == nil {
		return Endpoint{}, errNilResolver(replicateProviderName)
	}

	mapping, err := resolver.ResolveModel(params.Context, ResolveModelParams{
		Provider: replicateProviderName,
		Task:     task,
		Model:    params.Model,
	})
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    replicatePredictionRoute(mapping.ProviderModelID),
		Mapping: &mapping,
	}, nil
}

// replicatePredictionRoute returns Replicate's provider-native prediction
// route for the given provider model identifier. Version-pinned identifiers
// ("owner/name:version-hash") submit through the generic predictions
// endpoint with the version hash carried in the payload; plain identifiers
// address the model-scoped endpoint, exactly as both official Hugging Face
// clients route them.
func replicatePredictionRoute(providerModelID string) string {
	if _, version, found := strings.Cut(providerModelID, ":"); found && version != "" {
		return replicateProviderName + "/v1/predictions"
	}

	return replicateProviderName + "/v1/models/" + providerModelID + "/predictions"
}

// ReplicateTextToImageMixin provides the Replicate endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model
// ID to Replicate's provider model identifier at endpoint-construction
// time.
type ReplicateTextToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to Replicate's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// Replicate's identifier and returns the router-proxied prediction route
// for it.
func (m ReplicateTextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveReplicateEndpoint(m.Resolver, params, TaskTextToImage)
}

// TextToImageCodec returns the text-to-image codec, which encodes requests
// as a Replicate prediction and completes Replicate's asynchronous
// prediction exchange (wait, poll, download) before returning the generated
// image bytes. The codec adapts the payload to adapter models with the
// mapping resolved at endpoint-construction time, falling back to the
// mixin's resolver when used standalone (see
// [replicateInjectAdapterLora]).
func (m ReplicateTextToImageMixin) TextToImageCodec() TextToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return replicateTextToImageCodec{replicateAsyncCodec{queueTaskCodec{
		defaults:    replicateDefaultPollTiming(),
		mediaPrefix: replicateImageMediaPrefix,
		resolver:    m.Resolver,
	}}}
}

// ReplicateImageToImageMixin provides the Replicate endpoint and codec for
// image-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// image-to-image. Its [ModelResolver] maps the configured Hugging Face
// model ID to Replicate's provider model identifier at
// endpoint-construction time.
type ReplicateImageToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to Replicate's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// ImageToImageEndpoint resolves the configured Hugging Face model ID to
// Replicate's identifier and returns the router-proxied prediction route
// for it.
func (m ReplicateImageToImageMixin) ImageToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveReplicateEndpoint(m.Resolver, params, TaskImageToImage)
}

// ImageToImageCodec returns the image-to-image codec, which encodes requests
// as a Replicate prediction — the input image rides on all four keys
// different Replicate models expect — and completes Replicate's
// asynchronous prediction exchange before returning the generated image
// bytes. The codec adapts the payload to adapter models with the mapping
// resolved at endpoint-construction time, falling back to the mixin's
// resolver when used standalone (see [replicateInjectAdapterLora]).
func (m ReplicateImageToImageMixin) ImageToImageCodec() ImageToImageCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return replicateImageToImageCodec{replicateAsyncCodec{queueTaskCodec{
		defaults:    replicateDefaultPollTiming(),
		mediaPrefix: replicateImageMediaPrefix,
		resolver:    m.Resolver,
	}}}
}

// ReplicateTextToVideoMixin provides the Replicate endpoint and codec for
// text-to-video generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-video. Its [ModelResolver] maps the configured Hugging Face model
// ID to Replicate's provider model identifier at endpoint-construction
// time.
type ReplicateTextToVideoMixin struct {
	// Resolver maps the configured Hugging Face model ID to Replicate's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// TextToVideoEndpoint resolves the configured Hugging Face model ID to
// Replicate's identifier and returns the router-proxied prediction route
// for it.
func (m ReplicateTextToVideoMixin) TextToVideoEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveReplicateEndpoint(m.Resolver, params, TaskTextToVideo)
}

// TextToVideoCodec returns the text-to-video codec, which encodes requests
// as a Replicate prediction and completes Replicate's asynchronous
// prediction exchange (wait, poll, download) before returning the generated
// video bytes. The codec adapts the payload to adapter models with the
// mapping resolved at endpoint-construction time, falling back to the
// mixin's resolver when used standalone (see
// [replicateInjectAdapterLora]).
func (m ReplicateTextToVideoMixin) TextToVideoCodec() TextToVideoCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs select the default schedule.
	return replicateTextToVideoCodec{replicateAsyncCodec{queueTaskCodec{
		defaults:    replicateDefaultPollTiming(),
		mediaPrefix: replicateVideoMediaPrefix,
		resolver:    m.Resolver,
	}}}
}

// ReplicateSpeechRecognitionMixin provides the Replicate endpoint
// and codec for automatic speech recognition. It is a mixin: embed it in a
// type that also implements Provider.Name and Provider.ProviderSuffix to
// serve only speech recognition. Its [ModelResolver] maps the configured
// Hugging Face model ID to Replicate's provider model identifier at
// endpoint-construction time.
type ReplicateSpeechRecognitionMixin struct {
	// Resolver maps the configured Hugging Face model ID to Replicate's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// SpeechRecognitionEndpoint resolves the configured Hugging Face model ID
// to Replicate's identifier and returns the router-proxied prediction route
// for it.
func (m ReplicateSpeechRecognitionMixin) SpeechRecognitionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveReplicateEndpoint(m.Resolver, params, TaskAutomaticSpeechRecognition)
}

// SpeechRecognitionCodec returns the speech-recognition codec, which
// encodes requests as a Replicate prediction (the input audio rides on
// "audio" as a data: URI or URL) and completes Replicate's asynchronous
// prediction exchange before returning the transcript.
func (m ReplicateSpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	//nolint:exhaustruct_v5 // Zero polling knobs and media prefix are unused.
	return replicateSpeechRecognitionCodec{replicateAsyncCodec{queueTaskCodec{
		defaults: replicateDefaultPollTiming(),
		resolver: m.Resolver,
	}}}
}

// replicateAsyncCodec is Replicate's queue task codec: the shared
// [queueTaskCodec] base (resolver, polling knobs, media prefix, timing, and
// media decoding) plus Replicate's prediction exchange steps. Codec structs
// embed it so their PostSubmit and Decode implementations share one
// exchange; only the payload assembly and typed response construction
// differ per task.
type replicateAsyncCodec struct {
	queueTaskCodec
}

// postSubmitPrediction runs the shared Replicate exchange up to the
// completed prediction: decode the submit response, wait for — or poll —
// the prediction until it reaches a terminal status. The caller extracts
// the task-specific output from the result.
func (c replicateAsyncCodec) postSubmitPrediction(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
) (*replicatePrediction, error) {
	submit, err := replicateDecodePrediction(body)
	if err != nil {
		return nil, err
	}

	return c.awaitPrediction(ctx, params, &submit)
}

// postSubmitMedia runs the shared Replicate media exchange: complete the
// prediction, extract the first output URL, and download it. It returns the
// raw media bytes, the download's response headers, and provider metadata
// extracted from the completed prediction; the caller validates the output
// and builds the typed response via its own Decode.
func (c replicateAsyncCodec) postSubmitMedia(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
) ([]byte, http.Header, map[string]any, error) {
	pred, err := c.postSubmitPrediction(ctx, params, body)
	if err != nil {
		return nil, nil, nil, err
	}

	outputURL, err := replicateMediaOutputURL(pred)
	if err != nil {
		return nil, nil, nil, err
	}

	media, headers, err := queueDownloadOutput(ctx, params, outputURL)
	if err != nil {
		return nil, nil, nil, err
	}

	return media, headers, replicateMetadata(pred), nil
}

// replicateMetrics holds the completed prediction's timing breakdown.
type replicateMetrics struct {
	PredictTime *float64 `json:"predict_time"`
}

// replicatePrediction is the prediction object Replicate returns on submit
// and on every poll. Output stays absent (or JSON null) while the prediction
// runs; with the "Prefer: wait" submit header it is usually already present
// on the submit response.
type replicatePrediction struct {
	ID        string                  `json:"id"`
	Model     string                  `json:"model"`
	Status    string                  `json:"status"`
	CreatedAt string                  `json:"created_at"`
	Output    json.RawMessage         `json:"output"`
	Error     *string                 `json:"error"`
	Logs      string                  `json:"logs"`
	Metrics   *replicateMetrics       `json:"metrics"`
	URLs      replicatePredictionURLs `json:"urls"`
}

// replicatePredictionURLs holds the prediction's URL set. Get is the
// provider-native prediction URL polling reads the status from, e.g.
// "https://api.replicate.com/v1/predictions/{id}".
type replicatePredictionURLs struct {
	Get string `json:"get"`
}

// replicateDecodePrediction decodes a Replicate prediction JSON body. Any
// JSON body is accepted — the exchange's error paths surface through status
// classification and poll-target construction rather than here.
func replicateDecodePrediction(body []byte) (replicatePrediction, error) {
	var pred replicatePrediction
	if err := json.Unmarshal(body, &pred); err != nil {
		return replicatePrediction{}, queueDecodeTaskError("Replicate prediction response", err)
	}

	return pred, nil
}

// replicateHasOutput reports whether the prediction's output field carries a
// value. It is absent or JSON null while a prediction runs.
func replicateHasOutput(pred *replicatePrediction) bool {
	return len(pred.Output) > 0 && !bytes.Equal(pred.Output, []byte("null"))
}

// replicateTerminalPrediction classifies a decoded prediction. done reports
// whether the prediction reached a terminal status: true with a nil error
// means it succeeded (out holds the prediction whose output is extractable);
// true with an error means it failed (err describes the failure); false
// means the prediction is still starting or running — including unknown
// statuses — so polling should continue. A nil out is never returned with a
// nil error.
func replicateTerminalPrediction(
	pred *replicatePrediction,
) (out *replicatePrediction, done bool, err error) {
	switch pred.Status {
	case replicateStatusSucceeded:
		if !replicateHasOutput(pred) {
			return nil, true, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: replicateErrSucceededWithoutOutput,
				Err:     nil,
			}
		}

		return pred, true, nil
	case replicateStatusFailed, replicateStatusCanceled:
		return nil, true, replicatePredictionFailedError(pred)
	default:
		// "starting", "processing", and any status Replicate introduces
		// later: keep polling so new intermediate states do not abort
		// long-running predictions.
		return nil, false, nil
	}
}

// replicatePredictionFailedError builds the error surfaced for a
// terminally failed prediction, carrying the provider's error message and
// the prediction ID when present (see [queueTaskFailedError]).
func replicatePredictionFailedError(pred *replicatePrediction) error {
	message := "Replicate prediction failed"
	if pred.Error != nil && *pred.Error != "" {
		message += ": " + *pred.Error
	}
	if pred.ID != "" {
		message += " (prediction id " + pred.ID + ")"
	}

	return queueTaskFailedError(message)
}

// replicateMetadata extracts the provider-specific response metadata exposed
// on the typed responses' ProviderMetadata field. It returns nil when the
// completed prediction carries nothing useful, so callers never observe an
// empty map.
func replicateMetadata(pred *replicatePrediction) map[string]any {
	meta := make(map[string]any)
	if pred.ID != "" {
		meta["id"] = pred.ID
	}
	if pred.Model != "" {
		meta["model"] = pred.Model
	}
	if pred.Status != "" {
		meta["status"] = pred.Status
	}
	if pred.CreatedAt != "" {
		meta["created_at"] = pred.CreatedAt
	}
	if pred.Logs != "" {
		meta["logs"] = pred.Logs
	}
	if pred.Metrics != nil && pred.Metrics.PredictTime != nil {
		meta["metrics"] = map[string]any{"predict_time": *pred.Metrics.PredictTime}
	}
	if len(meta) == 0 {
		return nil
	}

	return meta
}

// replicatePollTarget resolves the prediction's poll URL and builds the
// request params for one poll. Polling always targets the host the submit
// went through, so the caller's bearer token applies; the resolution and
// request assembly are shared with the other queue providers (see
// [queuePollTarget]). Output downloads (see [queueDownloadOutput]) target
// Replicate's CDN and are sent unauthenticated.
func replicatePollTarget(
	params PostSubmitParams,
	pred *replicatePrediction,
) (request.Params, string, error) {
	if pred.URLs.Get == "" {
		return request.Params{}, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "Replicate prediction response has no prediction URL",
			Err:     nil,
		}
	}

	return queuePollTarget(params, pred.URLs.Get, replicateRouterPrefix, "Replicate prediction URL")
}

// awaitPrediction polls the prediction URL until the prediction completes,
// then returns the completed prediction. Callers must extract the output
// before using the result. The context bounds the whole exchange, including
// polling: a prediction that outlives the context surfaces the context
// error, and the prediction itself keeps running server-side.
func (c replicateAsyncCodec) awaitPrediction(
	ctx context.Context,
	params PostSubmitParams,
	submit *replicatePrediction,
) (*replicatePrediction, error) {
	if pred, done, taskErr := replicateTerminalPrediction(submit); done {
		return pred, taskErr
	}

	pollParams, pollPath, err := replicatePollTarget(params, submit)
	if err != nil {
		return nil, err
	}

	poll := func(ctx context.Context) (*replicatePrediction, time.Duration, error) {
		return replicatePollOnce(ctx, params, pollParams, pollPath)
	}

	return queueAwait(ctx, c.timing(), poll)
}

// replicatePollOnce issues one poll request and classifies the result. pred
// is non-nil when the prediction reached a terminal status; retryAfter is a
// server-provided wait override for the next round; any error stops
// polling. The exchange shape is shared with the other queue providers
// (see [queuePollOnce]).
func replicatePollOnce(
	ctx context.Context,
	params PostSubmitParams,
	pollParams request.Params,
	pollPath string,
) (*replicatePrediction, time.Duration, error) {
	return queuePollOnce(ctx, params, pollParams, pollPath, queueExchange[replicatePrediction]{
		decode:   replicateDecodePrediction,
		terminal: replicateTerminalPrediction,
	})
}

// replicateDownloadText downloads a provider-hosted text output (the ASR
// txt_file field) and returns its body as a string. It routes through
// [queueDownloadOutput]: the URL is a provider-owned output link, not a
// caller-authenticated follow-up.
func replicateDownloadText(
	ctx context.Context,
	params PostSubmitParams,
	textURL string,
) (string, error) {
	body, _, err := queueDownloadOutput(ctx, params, textURL)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

// replicateMediaOutputURL extracts the first output URL from a completed
// prediction's output, accepting Replicate's string and string-list output
// shapes. A missing, empty, or non-URL output is a serialization error.
func replicateMediaOutputURL(pred *replicatePrediction) (string, error) {
	if !replicateHasOutput(pred) {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: replicateErrSucceededWithoutOutput,
			Err:     nil,
		}
	}

	var outputURL string
	if err := json.Unmarshal(pred.Output, &outputURL); err == nil {
		if outputURL == "" {
			return "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "Replicate prediction output URL is empty",
				Err:     nil,
			}
		}

		return outputURL, nil
	}

	var outputURLs []string
	if err := json.Unmarshal(pred.Output, &outputURLs); err == nil {
		if len(outputURLs) == 0 || outputURLs[0] == "" {
			return "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "Replicate prediction output has no URL",
				Err:     nil,
			}
		}

		return outputURLs[0], nil
	}

	return "", &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: "Replicate prediction output is neither a URL nor a list of URLs",
		Err:     nil,
	}
}

// replicateAudioMIME labels sniffed audio bytes for the prediction's audio
// data URI, falling back to the official clients' default when the
// container is unrecognized. net/http's sniffer has no FLAC signature, so
// the shared FLAC check (see [hasFLACPrefix]) labels those bytes explicitly.
func replicateAudioMIME(raw []byte) string {
	mediaType := http.DetectContentType(raw)
	if strings.HasPrefix(mediaType, "audio/") {
		return mediaType
	}
	if hasFLACPrefix(raw) {
		return mimeTypeFLAC
	}

	return replicateAudioInputMIME
}

// replicateImageMIME labels sniffed image bytes for the prediction's image
// data URI, falling back to the official clients' default when the
// container is unrecognized.
func replicateImageMIME(raw []byte) string {
	mediaType := http.DetectContentType(raw)
	if strings.HasPrefix(mediaType, "image/") {
		return mediaType
	}

	return replicateImageInputMIME
}

// replicateInputDataURL passes self-describing inputs through unchanged and
// wraps every other input as a data URI labeled by the sniff function,
// matching the official Hugging Face clients' payload preparation. The
// canonical request DTOs carry base64 strings rather than media types, so
// the label is derived from the bytes themselves.
func replicateInputDataURL(input string, sniff func([]byte) string) (string, error) {
	if inputIsSelfDescribing(input) {
		return input, nil
	}

	raw, err := base64.StdEncoding.DecodeString(input)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeBase64Input,
			Err:     err,
		}
	}

	return "data:" + sniff(raw) + ";base64," + input, nil
}

// replicateAudioDataURL wraps a speech-recognition audio input as a data
// URI (see [replicateInputDataURL]).
func replicateAudioDataURL(input string) (string, error) {
	return replicateInputDataURL(input, replicateAudioMIME)
}

// replicateImageDataURL wraps an image-to-image input as a data URI (see
// [replicateInputDataURL]).
func replicateImageDataURL(input string) (string, error) {
	return replicateInputDataURL(input, replicateImageMIME)
}

// replicateInjectAdapterLora mirrors the official Hugging Face clients'
// LoRA handling for adapter models: when the model's Hub mapping declares a
// LoRA adapter (a fine-tune applied on top of a provider-hosted base model),
// the prediction's input gains a "lora_weights" entry pointing at the
// adapter's Hugging Face repository so Replicate loads the adapter weights.
// Without it the request succeeds but runs the base model. Both official
// clients apply this only on the image tasks — text-to-image and
// image-to-image — so replicateEncodePayload gates the injection on those.
//
// The injected entry names the Hugging Face repository root — not the
// weights file's resolve URL — exactly as both official clients send it;
// Replicate resolves the repository's weights itself. A caller-supplied
// "lora_weights" (via ProviderParams, merged before this call) wins.
func replicateInjectAdapterLora(mapping *ModelMapping, hfModel string, input map[string]any) {
	if mapping.Adapter != adapterLora || mapping.AdapterWeightsPath == "" {
		return
	}
	if _, ok := input[replicateLoraWeightsKey]; ok {
		return
	}

	input[replicateLoraWeightsKey] = DefaultHubBaseURL + "/" + hfModel
}

// replicateEncodePayload assembles the prediction payload around the
// task's input object: adapter models gain their "lora_weights" entry on
// the tasks where the official clients inject it (text-to-image and
// image-to-image), and version-pinned mappings (provider identifiers
// containing ":") carry the version hash on the payload's "version" key,
// matching the route the endpoint selected.
//
// The mapping resolved while constructing the endpoint (see
// [Endpoint.Mapping]) wins; otherwise the given resolver runs the lookup,
// mirroring WaveSpeed's standalone-codec fallback. A nil resolver with no
// supplied mapping leaves the payload unadapted — the pipeline always
// supplies a mapping, so this only affects standalone codec use.
func replicateEncodePayload(
	ctx context.Context,
	mapping *ModelMapping,
	resolver ModelResolver,
	task Task,
	hfModel string,
	input map[string]any,
) (map[string]any, error) {
	resolved := mapping
	if resolved == nil && resolver != nil {
		resolvedMapping, err := resolver.ResolveModel(ctx, ResolveModelParams{
			Provider: replicateProviderName,
			Task:     task,
			Model:    hfModel,
		})
		if err != nil {
			return nil, err
		}

		resolved = &resolvedMapping
	}

	payload := map[string]any{replicateInputKey: input}
	if resolved == nil {
		return payload, nil
	}

	// The official Hugging Face clients inject "lora_weights" only for the
	// image tasks: huggingface.js on text-to-image and image-to-image,
	// huggingface_hub's Replicate handler on text-to-image alone. This
	// matches the broader of the two — the narrower one's stricter behavior
	// only matters if the Hub tags a video or speech mapping with a LoRA
	// adapter, which today is image-model-only territory — and keeps
	// non-image payloads byte-identical to upstream even in that case:
	// Replicate models reject unknown input fields.
	if task == TaskTextToImage || task == TaskImageToImage {
		replicateInjectAdapterLora(resolved, hfModel, input)
	}
	if _, version, found := strings.Cut(resolved.ProviderModelID, ":"); found && version != "" {
		payload[replicateVersionKey] = version
	}

	return payload, nil
}

// replicateEncodeHeaders returns the request headers shared by Replicate
// encoders: a JSON body whose submit response is always JSON, plus the
// "Prefer: wait" directive asking Replicate to hold the request open until
// the prediction finishes (bounded by the provider's server-side wait
// window; the codec polls past it).
func replicateEncodeHeaders() http.Header {
	headers := jsonRequestHeaders(mimeTypeJSON)
	headers.Set(replicatePreferKey, replicatePreferWait)

	return headers
}

// replicateTextToImageCodec is a [Codec] for text-to-image that encodes
// canonical requests into a Replicate prediction and completes Replicate's
// asynchronous prediction exchange (wait, poll, download) before returning
// the generated image bytes.
type replicateTextToImageCodec struct {
	replicateAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToImageRequest,
		hftypes.TextToImageResponse,
	] = (*replicateTextToImageCodec)(nil)
	_ TextToImageCodec = (*replicateTextToImageCodec)(nil)
)

func (c replicateTextToImageCodec) Encode(
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	input := map[string]any{replicatePromptKey: req.Input}
	if err := mergeRequestParameters(input, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}

	payload, err := replicateEncodePayload(
		params.Context, params.Mapping, c.resolver, TaskTextToImage, params.Model, input,
	)
	if err != nil {
		return nil, nil, err
	}

	return replicateMarshalPayload(payload)
}

func (c replicateTextToImageCodec) Decode(
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

func (c replicateTextToImageCodec) PostSubmit(
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

// replicateImageToImageCodec is a [Codec] for image-to-image that encodes
// canonical requests into a Replicate prediction — the input image rides on
// all four keys different Replicate models expect — and completes
// Replicate's asynchronous prediction exchange before returning the
// generated image bytes.
type replicateImageToImageCodec struct {
	replicateAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.ImageToImageRequest,
		hftypes.ImageToImageResponse,
	] = (*replicateImageToImageCodec)(nil)
	_ ImageToImageCodec = (*replicateImageToImageCodec)(nil)
)

func (c replicateImageToImageCodec) Encode(
	params EncodeParams[hftypes.ImageToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	image, err := replicateImageDataURL(req.Input)
	if err != nil {
		return nil, nil, err
	}

	// Different Replicate models expect the image under different keys, so
	// the input rides on all of them; a model that reads one ignores the
	// rest, exactly as the official Hugging Face clients send it.
	input := map[string]any{
		"image":        image,
		"images":       []string{image},
		"input_image":  image,
		"input_images": []string{image},
	}
	if err := mergeRequestParameters(input, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}

	payload, err := replicateEncodePayload(
		params.Context, params.Mapping, c.resolver, TaskImageToImage, params.Model, input,
	)
	if err != nil {
		return nil, nil, err
	}

	return replicateMarshalPayload(payload)
}

func (c replicateImageToImageCodec) Decode(
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

func (c replicateImageToImageCodec) PostSubmit(
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

// replicateTextToVideoCodec is a [Codec] for text-to-video that encodes
// canonical requests into a Replicate prediction and completes Replicate's
// asynchronous prediction exchange (wait, poll, download) before returning
// the generated video bytes.
type replicateTextToVideoCodec struct {
	replicateAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.TextToVideoRequest,
		hftypes.TextToVideoResponse,
	] = (*replicateTextToVideoCodec)(nil)
	_ TextToVideoCodec = (*replicateTextToVideoCodec)(nil)
)

func (c replicateTextToVideoCodec) Encode(
	params EncodeParams[hftypes.TextToVideoRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	input := map[string]any{replicatePromptKey: req.Input}
	if err := mergeRequestParameters(input, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}

	payload, err := replicateEncodePayload(
		params.Context, params.Mapping, c.resolver, TaskTextToVideo, params.Model, input,
	)
	if err != nil {
		return nil, nil, err
	}

	return replicateMarshalPayload(payload)
}

func (c replicateTextToVideoCodec) Decode(
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

func (c replicateTextToVideoCodec) PostSubmit(
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

// replicateSpeechRecognitionCodec is a [Codec] for automatic speech
// recognition that encodes canonical requests into a Replicate prediction
// (the input audio rides on "audio" as a data: URI or URL) and completes
// Replicate's asynchronous prediction exchange before returning the
// transcript.
type replicateSpeechRecognitionCodec struct {
	replicateAsyncCodec
}

// Compile-time assertions that the codec participates in the async pipeline.
var (
	_ AsyncCodec[
		hftypes.SpeechRecognitionRequest,
		hftypes.SpeechRecognitionResponse,
	] = (*replicateSpeechRecognitionCodec)(nil)
	_ SpeechRecognitionCodec = (*replicateSpeechRecognitionCodec)(nil)
)

func (c replicateSpeechRecognitionCodec) Encode(
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	audio, err := replicateAudioDataURL(req.Input)
	if err != nil {
		return nil, nil, err
	}

	input := map[string]any{replicateAudioKey: audio}
	if err := mergeRequestParameters(input, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}

	payload, err := replicateEncodePayload(
		params.Context, params.Mapping, c.resolver,
		TaskAutomaticSpeechRecognition, params.Model, input,
	)
	if err != nil {
		return nil, nil, err
	}

	return replicateMarshalPayload(payload)
}

func (c replicateSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return replicateDecodeTranscript(params)
}

func (c replicateSpeechRecognitionCodec) PostSubmit(
	ctx context.Context,
	params PostSubmitParams,
	body []byte,
	_ http.Header,
) (hftypes.SpeechRecognitionResponse, error) {
	pred, err := c.postSubmitPrediction(ctx, params, body)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	text, err := replicateTranscript(ctx, params, pred)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	return hftypes.SpeechRecognitionResponse{
		Text:             text,
		Chunks:           nil, // Replicate output carries no chunk timestamps
		ProviderMetadata: replicateMetadata(pred),
	}, nil
}

// replicateMarshalPayload marshals a prediction payload into its request
// body and headers.
func replicateMarshalPayload(payload map[string]any) (body []byte, headers http.Header, err error) {
	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
	}

	return body, replicateEncodeHeaders(), nil
}

// replicateTranscript extracts the transcript from a completed
// speech-recognition prediction. Replicate models return the transcript as
// the prediction's output string, as a one-element list of that string, or
// — for multilingual models — as an object carrying "transcription",
// "translation", or a "txt_file" output URL whose text is downloaded. The
// extraction accepts all three shapes, matching the official Hugging Face
// clients; a transcription wins over a translation when both are present,
// as huggingface.js orders them. The caller passes a prediction whose
// output presence [replicateTerminalPrediction] already validated, so an
// output is always present here.
func replicateTranscript(
	ctx context.Context,
	params PostSubmitParams,
	pred *replicatePrediction,
) (string, error) {
	var transcript string
	if err := json.Unmarshal(pred.Output, &transcript); err == nil {
		return transcript, nil
	}

	var outputs []json.RawMessage
	if err := json.Unmarshal(pred.Output, &outputs); err == nil && len(outputs) > 0 {
		return replicateTranscriptValue(ctx, params, outputs[0])
	}

	return replicateTranscriptValue(ctx, params, pred.Output)
}

// replicateTranscriptValue extracts the transcript from one speech
// recognition output value: a string, or an object carrying a transcription,
// a translation, or a txt_file URL to download.
func replicateTranscriptValue(
	ctx context.Context,
	params PostSubmitParams,
	raw json.RawMessage,
) (string, error) {
	var transcript string
	if err := json.Unmarshal(raw, &transcript); err == nil {
		return transcript, nil
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "received malformed response from Replicate automatic-speech-recognition API",
			Err:     nil,
		}
	}

	for _, key := range []string{replicateTranscriptionKey, replicateTranslationKey} {
		if field, ok := fields[key]; ok {
			if err := json.Unmarshal(field, &transcript); err == nil {
				return transcript, nil
			}
		}
	}

	if field, ok := fields[replicateTxtFileKey]; ok {
		var txtFile string
		if err := json.Unmarshal(field, &txtFile); err == nil && txtFile != "" {
			return replicateDownloadText(ctx, params, txtFile)
		}
	}

	return "", &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: "received malformed response from Replicate automatic-speech-recognition API",
		Err:     nil,
	}
}

// replicateDecodeTranscript decodes a canonical speech-recognition response
// body — the shape [replicateSpeechRecognitionCodec.PostSubmit] builds its
// response from — into the typed response. A body without a string "text"
// field is a serialization error.
func replicateDecodeTranscript(params DecodeParams) (hftypes.SpeechRecognitionResponse, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return hftypes.SpeechRecognitionResponse{}, errUnexpectedContentType(contentType)
	}

	var fields struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(params.Body, &fields); err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	return hftypes.SpeechRecognitionResponse{
		Text:             fields.Text,
		Chunks:           nil,
		ProviderMetadata: nil,
	}, nil
}
