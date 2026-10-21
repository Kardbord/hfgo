package hftypes

import "slices"

// FeatureExtractionPooledEmbedding is a single pooled embedding vector.
//
// Use this with sentence-transformers-style models that reduce the model output
// to one vector for the whole input. For models that return one vector per
// token, use [FeatureExtractionTokenEmbeddings] instead.
//
// FeatureExtractionPooledEmbedding is a defined slice type. It can be assigned
// to and from []float64 without an explicit conversion.
type FeatureExtractionPooledEmbedding []float64

// Dimension returns the number of values in the embedding vector.
func (e FeatureExtractionPooledEmbedding) Dimension() int {
	return len(e)
}

// Clone returns a copy of the embedding vector.
func (e FeatureExtractionPooledEmbedding) Clone() FeatureExtractionPooledEmbedding {
	return slices.Clone(e)
}

// FeatureExtractionTokenEmbeddings contains one embedding vector per token for
// a single input.
//
// Use this with transformers-style models such as BART that return hidden states
// for every token. Each outer element corresponds to one token, and each inner
// slice is the hidden-state vector for that token.
//
// FeatureExtractionTokenEmbeddings is a defined slice type. It can be assigned
// to and from [][]float64 without an explicit conversion. A batch of
// token-level embeddings is represented as []FeatureExtractionTokenEmbeddings;
// convert each element when interoperating with APIs that require raw slices.
type FeatureExtractionTokenEmbeddings [][]float64

// Dimension returns the hidden size of the token embeddings. It returns 0 when
// there are no tokens.
//
// All token embeddings for a given model share the same hidden size; Dimension
// reports the size of the first token embedding.
func (e FeatureExtractionTokenEmbeddings) Dimension() int {
	if len(e) == 0 {
		return 0
	}

	return len(e[0])
}

// Clone returns a deep copy of the token embeddings.
//
// Each inner slice is copied, so mutations to the clone do not affect the
// original.
func (e FeatureExtractionTokenEmbeddings) Clone() FeatureExtractionTokenEmbeddings {
	if e == nil {
		return nil
	}

	out := make(FeatureExtractionTokenEmbeddings, len(e))
	for i := range e {
		out[i] = slices.Clone(e[i])
	}

	return out
}

// FeatureExtractionRequest represents a feature extraction
// inference request to the API for a single input.
type FeatureExtractionRequest struct {
	// The text to extract features from.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for feature extraction.
	Parameters *FeatureExtractionParameters `json:"parameters,omitempty"`
}

// FeatureExtractionBatchRequest represents a batched feature extraction
// inference request to the API for multiple inputs.
//
// NOTE: This is one of the few instances where batched inference is supported by
// the upstream API, AND it is officially documented; if abundantly cautious, consider
// that behavior may change without notice.
type FeatureExtractionBatchRequest struct {
	// The texts to extract features from.
	// Required.
	Inputs []string `json:"inputs"`

	// Additional inference parameters for feature extraction.
	Parameters *FeatureExtractionParameters `json:"parameters,omitempty"`
}

// FeatureExtractionParameters specify additional inference
// parameters for feature extraction tasks.
type FeatureExtractionParameters struct {
	// Whether to normalize the embeddings to unit L2 norm.
	Normalize *bool `json:"normalize,omitempty"`

	// The name of the prompt that should be used for encoding.
	// Must be a key in the sentence-transformers configuration
	// prompts dictionary.
	PromptName *string `json:"prompt_name,omitempty"`

	// Whether to truncate the input to the model's maximum length.
	Truncate *bool `json:"truncate,omitempty"`

	// The direction to truncate from when truncation is enabled.
	// Possible values: "left", "right".
	TruncationDirection *string `json:"truncation_direction,omitempty"`
}

const (
	// FeatureExtractionTruncationLeft truncates from the left (beginning) of the input.
	FeatureExtractionTruncationLeft = "left"
	// FeatureExtractionTruncationRight truncates from the right (end) of the input.
	FeatureExtractionTruncationRight = "right"
)

// Clone returns a deep defensive copy of the request.
func (r *FeatureExtractionRequest) Clone() FeatureExtractionRequest {
	if r == nil {
		return FeatureExtractionRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*FeatureExtractionParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the request.
func (r *FeatureExtractionBatchRequest) Clone() FeatureExtractionBatchRequest {
	if r == nil {
		return FeatureExtractionBatchRequest{}
	}
	out := *r
	out.Inputs = slices.Clone(r.Inputs)
	out.Parameters = cloneStructPtr(r.Parameters, (*FeatureExtractionParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *FeatureExtractionParameters) Clone() FeatureExtractionParameters {
	if p == nil {
		return FeatureExtractionParameters{}
	}
	out := *p
	out.Normalize = clonePtr(p.Normalize)
	out.PromptName = clonePtr(p.PromptName)
	out.Truncate = clonePtr(p.Truncate)
	out.TruncationDirection = clonePtr(p.TruncationDirection)

	return out
}
