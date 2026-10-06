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
	Decode(params DecodeParams) (Resp, error)
}

type (
	// ChatCodec is the Codec for the Chat task.
	ChatCodec = Codec[hftypes.ChatRequest, hftypes.ChatResponse]

	// ChatStreamCodec is the Codec for the ChatStream task.
	ChatStreamCodec = Codec[hftypes.ChatRequest, hftypes.ChatStreamResponse]

	// FeatureExtractionBatchCodec is the Codec for the FeatureExtractBatch task.
	FeatureExtractionBatchCodec = Codec[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction]

	// FeatureExtractionCodec is the Codec for the FeatureExtract task.
	FeatureExtractionCodec = Codec[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction]

	// FillMaskBatchCodec is the Codec for the FillMaskBatch task.
	FillMaskBatchCodec = Codec[hftypes.FillMaskBatchRequest, [][]hftypes.FillMaskPrediction]

	// FillMaskCodec is the Codec for the FillMask task.
	FillMaskCodec = Codec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]

	// QuestionAnsweringCodec is the Codec for the AnswerQuestion task.
	QuestionAnsweringCodec = Codec[hftypes.QuestionAnsweringRequest, []hftypes.QuestionAnswering]

	// SummarizationBatchCodec is the Codec for the SummarizeBatch task.
	SummarizationBatchCodec = Codec[hftypes.SummarizationBatchRequest, []hftypes.Summarization]

	// SummarizationCodec is the Codec for the Summarize task.
	SummarizationCodec = Codec[hftypes.SummarizationRequest, []hftypes.Summarization]

	// TableQuestionAnsweringCodec is the Codec for the AnswerTableQuestion task.
	TableQuestionAnsweringCodec = Codec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]

	// TextClassificationBatchCodec is the Codec for the ClassifyTextBatch task.
	TextClassificationBatchCodec = Codec[hftypes.TextClassificationBatchRequest, [][]hftypes.TextClassification]

	// TextClassificationCodec is the Codec for the ClassifyText task.
	TextClassificationCodec = Codec[hftypes.TextClassificationRequest, []hftypes.TextClassification]

	// TokenClassificationBatchCodec is the Codec for the ClassifyTokensBatch task.
	TokenClassificationBatchCodec = Codec[hftypes.TokenClassificationBatchRequest, [][]hftypes.TokenClassification]

	// TokenClassificationCodec is the Codec for the ClassifyTokens task.
	TokenClassificationCodec = Codec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]

	// TranslationBatchCodec is the Codec for the TranslateBatch task.
	TranslationBatchCodec = Codec[hftypes.TranslationBatchRequest, []hftypes.Translation]

	// TranslationCodec is the Codec for the Translate task.
	TranslationCodec = Codec[hftypes.TranslationRequest, []hftypes.Translation]

	// ZeroShotTextClassificationBatchCodec is the Codec for the ZeroShotClassifyTextBatch task.
	ZeroShotTextClassificationBatchCodec = Codec[hftypes.ZeroShotTextClassificationBatchRequest, []hftypes.ZeroShotTextClassificationBatched]

	// ZeroShotTextClassificationCodec is the Codec for the ZeroShotClassifyText task.
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
// of kind SDKErrorKindSerialization is returned. A serialization error is also
// returned if the body fails to unmarshal.
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
	if contentType == "" {
		return false
	}

	mediatype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}

	return mediatype == mimeTypeJSON || strings.HasSuffix(mediatype, "+json")
}
