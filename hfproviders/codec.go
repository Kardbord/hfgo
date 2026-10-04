package hfproviders

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// Codec knows how to transform canonical Hugging Face request types and provider wire types,
// as well as provider response wire types and canonical Hugging Face response types.
type Codec[Req, Resp any] interface {
	// Encode transforms a canonical Hugging Face request type into its provider-format
	// wire type.
	Encode(req Req) (providerBody []byte, providerContentType string, err error)

	// Decode transforms a provider wire type response into its canonical Hugging Face
	// response type.
	Decode(resp []byte, providerContentType string) (Resp, error)
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
	ZeroShotTextClassificationBatchCodec = Codec[hftypes.ZeroShotTextClassificationBatchRequest, [][]hftypes.ZeroShotTextClassification]

	// ZeroShotTextClassificationCodec is the Codec for the ZeroShotClassifyText task.
	ZeroShotTextClassificationCodec = Codec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]
)

// JSONCodec is a generic Codec implementation that encodes requests and decodes
// responses as JSON.
type JSONCodec[Req, Resp any] struct{}

// Encode marshals req into a JSON request body and returns it along with the
// "application/json" content type. It returns an *hferrors.SDKError of kind
// SDKErrorKindSerialization if marshaling fails.
func (JSONCodec[Req, Resp]) Encode(req Req) (body []byte, contentType string, err error) {
	contentType = "application/json"
	body, err = json.Marshal(req)
	if err != nil {
		return nil, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to marshal request body",
			Err:     err,
		}
	}

	return body, contentType, nil
}

// Decode unmarshals a JSON response body into the response type. The contentType
// must begin with "application/json"; otherwise an *hferrors.SDKError of kind
// SDKErrorKindSerialization is returned. A serialization error is also returned
// if the body is empty or fails to unmarshal.
func (JSONCodec[Req, Resp]) Decode(respBody []byte, contentType string) (resp Resp, err error) {
	if !strings.HasPrefix(contentType, "application/json") {
		return resp, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got" + contentType,
			Err:     nil,
		}
	}

	if err := json.Unmarshal(respBody, &resp); err != nil {
		if errors.Is(err, io.EOF) {
			return resp, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "empty response body",
				Err:     err,
			}
		}

		return resp, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode response body",
			Err:     err,
		}
	}

	return resp, nil
}
