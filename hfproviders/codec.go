package hfproviders

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"mime"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const (
	errFailedToDecodeResponseBody = "failed to decode response body"
	// errFailedToMarshalRequestBody is the message surfaced when a request
	// body cannot be marshaled.
	errFailedToMarshalRequestBody = "failed to marshal request body"
	// errFailedToDecodeBase64Input is the message surfaced when a request's
	// base64 payload (e.g. speech-recognition audio) cannot be decoded.
	errFailedToDecodeBase64Input = "failed to decode base64 input"
	// errFailedToMarshalInferenceParameters is the message surfaced when a
	// typed inference-parameters struct cannot be marshaled (e.g. it carries
	// a value JSON cannot represent).
	errFailedToMarshalInferenceParameters = "failed to marshal inference parameters"
	// errFailedToFlattenInferenceParameters is the message surfaced when a
	// marshaled inference-parameters struct cannot be flattened back into a
	// JSON object map.
	errFailedToFlattenInferenceParameters = "failed to flatten inference parameters"
	// errFailedToDecodeRequestBody is the message surfaced when an already
	// marshaled request body cannot be decoded back into a JSON object map
	// (e.g. by codecs that post-process the body, such as the
	// [hfParamsCodec] ProviderParams merge or Together's response-format
	// flattening).
	errFailedToDecodeRequestBody = "failed to decode request body"
	mimeTypeJSON                 = "application/json"
	mimeTypeAny                  = "*/*"
	mimeTypeFLAC                 = "audio/flac"
	// headerAccept is the Accept request header name, used by every encoder
	// that describes its wire format through [jsonRequestHeaders] and by the
	// queue providers' follow-up requests.
	headerAccept = "Accept"
)

// adapterLora is the Hub mapping adapter value marking a model that ships
// LoRA weights applied on top of a provider-hosted base model. The
// media-output providers key their LoRA payload injection on it (see
// [ModelMapping.Adapter]).
const adapterLora = "lora"

// flacMagic is the magic prefix of a FLAC stream. net/http's byte sniffer
// has no FLAC signature, so the media-type sniffers in this package layer an
// explicit [hasFLACPrefix] check on top of the sniffer's verdict.
const flacMagic = "fLaC"

// hasFLACPrefix reports whether raw begins with the FLAC stream magic.
func hasFLACPrefix(raw []byte) bool {
	return bytes.HasPrefix(raw, []byte(flacMagic))
}

// huggingfaceRouterHost is the Hugging Face router host. Providers that
// resolve router-proxied follow-up URLs compare the caller's base URL host
// against it.
const huggingfaceRouterHost = "router.huggingface.co"

// EncodeParams carries the inputs to [Codec.Encode]. The per-call context is
// passed to Encode separately.
type EncodeParams[Req any] struct {
	// Request is the canonical Hugging Face request payload to encode.
	Request Req

	// Model is the model identifier for this request, mirroring
	// [EndpointParams.Model]: the resolved routing ID (including any
	// provider suffix) for chat tasks, the configured model ID for
	// pipeline tasks. Wire formats that carry the model in the body for
	// pipeline tasks can populate it from here, since the canonical
	// request types do not include it.
	Model string

	// Mapping mirrors [Endpoint.Mapping]: the model mapping the provider
	// resolved while constructing the endpoint, or nil when the provider
	// resolved none. Codecs that need mapping-driven payload changes (e.g.
	// adapter-model handling) can reuse it instead of looking the same
	// mapping up again through their resolver.
	Mapping *ModelMapping
}

// DecodeParams carries the inputs to [Codec.Decode]. The per-call context is
// passed to Decode separately.
type DecodeParams struct {
	// Body is the response body. It is never empty: the pipeline surfaces
	// an empty 2xx body as an error before Decode runs.
	Body []byte

	// Event is the SSE "event:" field for this payload. It is only set for
	// streaming decodes: unary responses leave it empty, and OpenAI-style
	// streams (data-only frames) do as well. Providers whose framing
	// carries meaning in the event name can dispatch on it, and may return
	// [hferrors.EndOfStreamError] or [hferrors.SkipEventError] from Decode
	// to end the stream or suppress the frame.
	Event string

	// Headers holds the response headers; read the body media type via
	// Headers.Get("Content-Type"). For SSE event payloads, Headers carries
	// only a synthesized Content-Type of "application/json" — event frames
	// have no headers of their own.
	Headers http.Header
}

// PostSubmitParams carries the transport configuration an [AsyncCodec] needs
// to finish a multi-step exchange after the initial submit response. The
// task pipeline populates it from the caller's [hfopts.Options], so follow-up
// requests use the same HTTP client, host, credentials, User-Agent, and
// response-size limits as the original request.
type PostSubmitParams struct {
	// HTTPClient is the caller-configured HTTP client. It is never nil when
	// dispatched by the task pipeline.
	HTTPClient *http.Client

	// BaseURL is the inference base URL the original request was sent to
	// (for example "https://router.huggingface.co"). Provider-relative
	// result paths are resolved against it.
	BaseURL string

	// Token is the caller's bearer token, sent as Authorization on
	// follow-up requests that target the provider host. The value is empty
	// when the caller configured no token.
	Token string

	// UserAgent is the User-Agent header value the original request was sent
	// with, reused on follow-up requests so the whole exchange is attributed
	// consistently. It is empty when the caller configured no User-Agent.
	UserAgent string

	// MaxResponseBodyBytes caps how much of any follow-up response body is
	// buffered at once. Values <= 0 mean "use the pipeline default".
	MaxResponseBodyBytes int64
}

// Codec knows how to transform canonical Hugging Face request types and provider wire types,
// as well as provider response wire types and canonical Hugging Face response types.
// Implementations must be safe for concurrent use.
type Codec[Req, Resp any] interface {
	// Encode transforms a canonical Hugging Face request type into its
	// provider-format wire type, returning the request body and any request
	// headers that describe the wire format (e.g. Content-Type, Accept).
	// Returned headers are applied as defaults: headers set by the caller
	// via hfopts options win. Implementations must not return an
	// Authorization header.
	Encode(
		ctx context.Context,
		params EncodeParams[Req],
	) (body []byte, headers http.Header, err error)

	// Decode transforms a provider wire type response into its canonical
	// Hugging Face response type.
	//
	// While streaming, Decode may steer the pipeline by returning
	// [hferrors.EndOfStreamError] to end the stream (the consumer observes
	// io.EOF) or [hferrors.SkipEventError] to suppress the current frame.
	// Any other non-nil error surfaces to the stream consumer as a
	// serialization failure. Termination is final: the terminating frame's
	// own value is not delivered, and frames the server sends after it are
	// discarded — anything a consumer receives precedes the terminal frame.
	// The control signals have no meaning for unary responses, where any
	// error is returned as-is.
	Decode(ctx context.Context, params DecodeParams) (Resp, error)
}

// AsyncCodec is an optional [Codec] extension for providers whose wire format
// needs follow-up HTTP requests after the initial submit response, such as
// the asynchronous task APIs some generation providers expose.
//
// Implementing providers keep the full exchange inside the codec: the task
// pipeline calls PostSubmit instead of Decode when the codec opts in, passing
// the submit response body and headers plus the transport configuration needed
// for follow-up requests. Everything else — polling intervals, terminal-state
// handling, output retrieval — stays provider-owned wire-format logic.
type AsyncCodec[Req, Resp any] interface {
	Codec[Req, Resp]

	// PostSubmit completes the exchange begun by Encode: body and headers
	// are the initial submit response, and params carries the transport
	// configuration for any follow-up requests. The context carries the
	// caller's deadline and cancellation signals and bounds the whole
	// exchange, including polling.
	PostSubmit(
		ctx context.Context,
		params PostSubmitParams,
		body []byte,
		headers http.Header,
	) (Resp, error)
}

type (
	// ChatCodec is the Codec for the Chat task.
	ChatCodec = Codec[hftypes.ChatRequest, hftypes.ChatResponse]

	// ChatStreamCodec is the Codec for the ChatStream task.
	ChatStreamCodec = Codec[hftypes.ChatRequest, hftypes.ChatStreamResponse]

	// ObjectDetectionCodec is the Codec for the ObjectDetection task.
	ObjectDetectionCodec = Codec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]

	// FeatureExtractionPooledBatchCodec is the Codec for pooled output from the
	// FeatureExtractionBatch task.
	FeatureExtractionPooledBatchCodec = Codec[
		hftypes.FeatureExtractionBatchRequest,
		[]hftypes.FeatureExtractionPooledEmbedding,
	]

	// FeatureExtractionPooledCodec is the Codec for pooled output from the
	// FeatureExtraction task.
	FeatureExtractionPooledCodec = Codec[
		hftypes.FeatureExtractionRequest,
		hftypes.FeatureExtractionPooledEmbedding,
	]

	// FeatureExtractionTokenBatchCodec is the Codec for token-level output from
	// the FeatureExtractionBatch task.
	FeatureExtractionTokenBatchCodec = Codec[
		hftypes.FeatureExtractionBatchRequest,
		[]hftypes.FeatureExtractionTokenEmbeddings,
	]

	// FeatureExtractionTokenCodec is the Codec for token-level output from the
	// FeatureExtraction task.
	FeatureExtractionTokenCodec = Codec[
		hftypes.FeatureExtractionRequest,
		hftypes.FeatureExtractionTokenEmbeddings,
	]

	// FillMaskCodec is the Codec for the FillMask task.
	FillMaskCodec = Codec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]

	// ImageClassificationCodec is the Codec for the ImageClassification task.
	ImageClassificationCodec = Codec[
		hftypes.ImageClassificationRequest,
		[]hftypes.ImageClassification,
	]

	// ImageSegmentationCodec is the Codec for the ImageSegmentation task.
	ImageSegmentationCodec = Codec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]

	// QuestionAnsweringCodec is the Codec for the QuestionAnswering task.
	QuestionAnsweringCodec = Codec[hftypes.QuestionAnsweringRequest, []hftypes.QuestionAnswer]

	// SummarizationCodec is the Codec for the Summarization task.
	SummarizationCodec = Codec[hftypes.SummarizationRequest, []hftypes.Summarization]

	// TableQuestionAnsweringCodec is the Codec for the TableQuestionAnswering task.
	TableQuestionAnsweringCodec = Codec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]

	// TextClassificationCodec is the Codec for the TextClassification task.
	TextClassificationCodec = Codec[hftypes.TextClassificationRequest, []hftypes.TextClassification]

	// ImageToImageCodec is the Codec for the ImageToImage task.
	ImageToImageCodec = Codec[hftypes.ImageToImageRequest, hftypes.ImageToImageResponse]

	// TextToImageCodec is the Codec for the TextToImage task.
	TextToImageCodec = Codec[hftypes.TextToImageRequest, hftypes.TextToImageResponse]

	// TextToVideoCodec is the Codec for the TextToVideo task.
	TextToVideoCodec = Codec[hftypes.TextToVideoRequest, hftypes.TextToVideoResponse]

	// TokenClassificationCodec is the Codec for the TokenClassification task.
	TokenClassificationCodec = Codec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]

	// TranslationCodec is the Codec for the Translation task.
	TranslationCodec = Codec[hftypes.TranslationRequest, []hftypes.Translation]

	// ZeroShotTextClassificationCodec is the Codec for the ZeroShotTextClassification task.
	ZeroShotTextClassificationCodec = Codec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]

	// SpeechRecognitionCodec is the Codec for the SpeechRecognition task.
	SpeechRecognitionCodec = Codec[hftypes.SpeechRecognitionRequest, hftypes.SpeechRecognitionResponse]
)

// jsonRequestHeaders returns the request headers shared by JSON-body request
// encoders: the body's Content-Type is application/json, and accept names the
// media type the response is expected to carry. Providers whose response
// subtype is provider-chosen (e.g. media bytes) pass [mimeTypeAny].
func jsonRequestHeaders(accept string) http.Header {
	headers := http.Header{}
	headers.Set("Content-Type", mimeTypeJSON)
	headers.Set(headerAccept, accept)

	return headers
}

// JSONCodec is a generic Codec implementation that encodes requests and decodes
// responses as JSON.
type JSONCodec[Req, Resp any] struct{}

// Encode marshals params.Request into a JSON request body and returns it along
// with Content-Type and Accept headers of "application/json". If marshaling
// returns an existing *hferrors.SDKError, such as a validation error from a
// custom MarshalJSON method, it is returned unchanged. Other marshal failures
// become an *hferrors.SDKError of kind SDKErrorKindSerialization.
func (JSONCodec[Req, Resp]) Encode(
	_ context.Context,
	params EncodeParams[Req],
) (body []byte, headers http.Header, err error) {
	body, err = json.Marshal(params.Request)
	if err != nil {
		if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
			return nil, nil, sdkErr
		}

		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, jsonRequestHeaders(mimeTypeJSON), nil
}

// Decode unmarshals a JSON response body into the response type. The body
// media type, read from params.Headers via the "Content-Type" header, must be
// application/json or a structured +json type; otherwise an *hferrors.SDKError
// of kind SDKErrorKindSerialization is returned. A missing or empty
// Content-Type fails the same way — the media type is never inferred from the
// body. Callers whose upstream omits Content-Type must synthesize
// "application/json" before delegating to JSONCodec. If unmarshaling returns an
// existing *hferrors.SDKError, such as a validation error from a custom
// UnmarshalJSON method, it is returned unchanged. Other unmarshal failures
// become an *hferrors.SDKError of kind SDKErrorKindSerialization.
func (JSONCodec[Req, Resp]) Decode(_ context.Context, params DecodeParams) (resp Resp, err error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return resp, errUnexpectedContentType(contentType)
	}

	if err := json.Unmarshal(params.Body, &resp); err != nil {
		if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
			return resp, sdkErr
		}
		if errors.Is(err, io.EOF) {
			return resp, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "empty response body",
				Err:     err,
			}
		}

		return resp, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	return resp, nil
}

// isJSONContentType reports whether contentType is application/json or a
// structured +json media type. Media-type parameters (e.g. charset) are
// ignored.
func isJSONContentType(contentType string) bool {
	mediatype, ok := parseMediaType(contentType)
	if !ok {
		return false
	}

	return mediatype == mimeTypeJSON || isStructuredJSONSuffix(mediatype)
}

// isStructuredJSONSuffix reports whether mediaType ends in the +json
// structured-syntax suffix with a proper type/subtype before it. RFC 6838
// requires the suffix to follow a subtype, so a bare "+json" (which
// mime.ParseMediaType tolerates) is not a JSON media type.
func isStructuredJSONSuffix(mediaType string) bool {
	base, ok := strings.CutSuffix(mediaType, "+json")

	return ok && strings.Contains(base, "/")
}

// parseMediaType parses contentType and returns its normalized media type (for
// example "application/json" or "image/png"), ignoring media-type parameters
// such as charset. It reports false when contentType is empty or malformed.
func parseMediaType(contentType string) (string, bool) {
	if contentType == "" {
		return "", false
	}

	mediatype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return "", false
	}

	return mediatype, true
}

// flattenParameters marshals a typed parameters struct into a JSON object map
// so its non-nil fields can be merged into a provider's flat request body.
// Nil parameters yield an empty map. Numbers are decoded as json.Number so
// their exact literal survives the marshal/unmarshal round trip; re-marshaling
// the map emits the original digits instead of float64 rounding artifacts for
// values beyond float64 precision.
func flattenParameters(parameters any) (map[string]any, error) {
	if parameters == nil {
		return map[string]any{}, nil
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalInferenceParameters,
			Err:     err,
		}
	}

	flat := map[string]any{}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&flat); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToFlattenInferenceParameters,
			Err:     err,
		}
	}

	return flat, nil
}

// mergeInferenceParameters flattens a request's typed inference parameters
// and merges them into payload. Nil parameters leave payload untouched; see
// [flattenParameters] for the flattening semantics.
func mergeInferenceParameters(payload map[string]any, parameters any) error {
	if parameters == nil {
		return nil
	}

	flat, err := flattenParameters(parameters)
	if err != nil {
		return err
	}
	maps.Copy(payload, flat)

	return nil
}

// mergeRequestParameters merges a request's typed inference parameters and
// its ProviderParams into payload, in that order: ProviderParams entries win
// over typed-parameter entries. This is the merge order every flat-wire
// provider encodes; callers with additional payload post-processing (size
// folds, forced keys) sequence it between their own steps.
func mergeRequestParameters(
	payload map[string]any,
	parameters any,
	providerParams map[string]any,
) error {
	if err := mergeInferenceParameters(payload, parameters); err != nil {
		return err
	}
	maps.Copy(payload, providerParams)

	return nil
}

// decodeBodyToPayloadMap decodes an already marshaled JSON request body into
// a payload map. Numbers are decoded as [json.Number] so a re-marshal of the
// map preserves the original numeric literals exactly (no float64
// round-tripping). Decode failures become an *hferrors.SDKError of kind
// SDKErrorKindSerialization.
func decodeBodyToPayloadMap(body []byte) (map[string]any, error) {
	payload := map[string]any{}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeRequestBody,
			Err:     err,
		}
	}

	return payload, nil
}

// mergeProviderParamsTopLevel shallow-copies a request's ProviderParams
// entries into the payload's top level. Existing payload entries with the
// same key are overridden, which is the documented ProviderParams contract:
// entries override typed fields. The providerParams map itself is treated as
// read-only and never mutated. Nil providerParams leave payload untouched.
func mergeProviderParamsTopLevel(payload, providerParams map[string]any) {
	maps.Copy(payload, providerParams)
}

// marshalRequestBody marshals a request payload map into its JSON request
// body, wrapping marshal failures in the package's shared serialization
// error.
func marshalRequestBody(payload map[string]any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return body, nil
}

// inputIsSelfDescribing reports whether input already carries its own
// addressing — a data URI or an absolute HTTP(S) URL — and so passes through
// provider payloads unchanged rather than being wrapped as a data URI. The
// queue-based media providers share this rule when preparing image, video,
// and audio inputs.
func inputIsSelfDescribing(input string) bool {
	return input == "" ||
		strings.HasPrefix(input, "data:") ||
		strings.HasPrefix(input, "http://") ||
		strings.HasPrefix(input, "https://")
}
