package hfproviders

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// encodeOpenAIEmbeddings builds the OpenAI-style embeddings JSON body shared
// by every provider in this package that speaks the OpenAI embeddings wire
// format (currently DeepInfra and Scaleway): "input" carries the text(s),
// provider parameters merge at the top level, and "model" is applied last
// from the resolved mapping so provider parameters cannot override it. The
// hf-inference feature-extraction parameters the canonical request types
// carry (e.g. normalize) are not forwarded — OpenAI-style embeddings APIs do
// not accept them; provider-specific parameters such as encoding_format ride
// ProviderParams instead. T mirrors the OpenAI input field's two shapes: a
// single text for the single-input task, a batch of texts for the batch task.
func encodeOpenAIEmbeddings[T string | []string](
	input T,
	providerParams map[string]any,
	mapping *ModelMapping,
) (body []byte, headers http.Header, err error) {
	payload := map[string]any{}
	for key, value := range providerParams {
		if value == nil {
			continue
		}

		payload[key] = value
	}
	payload["input"] = input
	payload["model"] = mapping.ProviderModelID

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
	}

	return body, jsonRequestHeaders(mimeTypeJSON), nil
}

// openAIEmbeddingEntry is one embedding in the embeddings endpoint's
// response.
type openAIEmbeddingEntry struct {
	Embedding []float64 `json:"embedding"`
}

// openAIEmbeddingsResponse is the embeddings endpoint's response shape: a
// data array holding one embedding per input, in order.
type openAIEmbeddingsResponse struct {
	Data []openAIEmbeddingEntry `json:"data"`
}

// decodeOpenAIEmbeddings decodes the OpenAI-style embeddings endpoint's
// response into pooled embeddings, one per data entry. A body without
// embeddings — or an entry without an embedding array — is a serialization
// error.
func decodeOpenAIEmbeddings(
	params DecodeParams,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, errUnexpectedContentType(contentType)
	}

	var resp openAIEmbeddingsResponse
	if err := json.Unmarshal(params.Body, &resp); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	if len(resp.Data) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "embeddings response contains no embeddings",
			Err:     nil,
		}
	}

	embeddings := make([]hftypes.FeatureExtractionPooledEmbedding, 0, len(resp.Data))
	for i, entry := range resp.Data {
		if entry.Embedding == nil {
			return nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf("embeddings response entry %d is missing an embedding", i),
				Err:     nil,
			}
		}

		embeddings = append(embeddings, hftypes.FeatureExtractionPooledEmbedding(entry.Embedding))
	}

	return embeddings, nil
}
