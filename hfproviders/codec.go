package hfproviders

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const (
	errFailedToDecodeResponseBody = "failed to decode response body"
	mimeTypeJSON                  = "application/json"
)

// EncodeParams carries the inputs to [Codec.Encode].
type EncodeParams[Req any] struct {
	// Context carries the caller's request context, including its deadline
	// and cancellation signals. It is never nil (it falls back to
	// context.Background()). Implementations must not retain it beyond the
	// duration of the call.
	//
	//nolint:containedctx // EncodeParams is a per-call carrier, not stored state.
	Context context.Context

	// Request is the canonical Hugging Face request payload to encode.
	Request Req

	// Model is the model identifier for this request, mirroring
	// [EndpointParams.Model]: the resolved routing ID (including any
	// provider suffix) for chat tasks, the configured model ID for
	// pipeline tasks. Wire formats that carry the model in the body for
	// pipeline tasks can populate it from here, since the canonical
	// request types do not include it.
	Model string
}

// DecodeParams carries the inputs to [Codec.Decode].
type DecodeParams struct {
	// Context carries the caller's request context, including its deadline
	// and cancellation signals. It is never nil (it falls back to
	// context.Background()). Implementations must not retain it beyond the
	// duration of the call.
	//
	//nolint:containedctx // DecodeParams is a per-call carrier, not stored state.
	Context context.Context

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
	Encode(params EncodeParams[Req]) (body []byte, headers http.Header, err error)

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
	Decode(params DecodeParams) (Resp, error)
}

type (
	// ChatCodec is the Codec for the Chat task.
	ChatCodec = Codec[hftypes.ChatRequest, hftypes.ChatResponse]

	// ChatStreamCodec is the Codec for the ChatStream task.
	ChatStreamCodec = Codec[hftypes.ChatRequest, hftypes.ChatStreamResponse]

	// ObjectDetectionCodec is the Codec for the ObjectDetection task.
	ObjectDetectionCodec = Codec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]

	// FeatureExtractionBatchCodec is the Codec for the FeatureExtractionBatch task.
	FeatureExtractionBatchCodec = Codec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]

	// FeatureExtractionCodec is the Codec for the FeatureExtraction task.
	FeatureExtractionCodec = Codec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]

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
	QuestionAnsweringCodec = Codec[hftypes.QuestionAnsweringRequest, []hftypes.QuestionAnswering]

	// SummarizationCodec is the Codec for the Summarization task.
	SummarizationCodec = Codec[hftypes.SummarizationRequest, []hftypes.Summarization]

	// TableQuestionAnsweringCodec is the Codec for the TableQuestionAnswering task.
	TableQuestionAnsweringCodec = Codec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]

	// TextClassificationCodec is the Codec for the TextClassification task.
	TextClassificationCodec = Codec[hftypes.TextClassificationRequest, []hftypes.TextClassification]

	// TextToImageCodec is the Codec for the TextToImage task.
	TextToImageCodec = Codec[hftypes.TextToImageRequest, hftypes.TextToImageResponse]

	// TokenClassificationCodec is the Codec for the TokenClassification task.
	TokenClassificationCodec = Codec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]

	// TranslationCodec is the Codec for the Translation task.
	TranslationCodec = Codec[hftypes.TranslationRequest, []hftypes.Translation]

	// ZeroShotTextClassificationCodec is the Codec for the ZeroShotTextClassification task.
	ZeroShotTextClassificationCodec = Codec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]
)

// JSONCodec is a generic Codec implementation that encodes requests and decodes
// responses as JSON.
type JSONCodec[Req, Resp any] struct{}

// Encode marshals params.Request into a JSON request body and returns it along
// with Content-Type and Accept headers of "application/json". It returns an
// *hferrors.SDKError of kind SDKErrorKindSerialization if marshaling fails.
func (JSONCodec[Req, Resp]) Encode(
	params EncodeParams[Req],
) (body []byte, headers http.Header, err error) {
	body, err = json.Marshal(params.Request)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to marshal request body",
			Err:     err,
		}
	}

	headers = http.Header{}
	headers.Set("Content-Type", mimeTypeJSON)
	headers.Set("Accept", mimeTypeJSON)

	return body, headers, nil
}

// Decode unmarshals a JSON response body into the response type. The body
// media type, read from params.Headers via the "Content-Type" header, must be
// application/json or a structured +json type; otherwise an *hferrors.SDKError
// of kind SDKErrorKindSerialization is returned. A missing or empty
// Content-Type fails the same way — the media type is never inferred from the
// body. Callers whose upstream omits Content-Type must synthesize
// "application/json" before delegating to JSONCodec. A serialization error is
// also returned if the body fails to unmarshal.
func (JSONCodec[Req, Resp]) Decode(params DecodeParams) (resp Resp, err error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return resp, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
	}

	if err := json.Unmarshal(params.Body, &resp); err != nil {
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

	return mediatype == mimeTypeJSON || strings.HasSuffix(mediatype, "+json")
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
