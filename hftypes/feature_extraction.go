package hftypes

import "slices"

// FeatureExtraction is a type alias for a single embedding vector
// returned by the feature extraction endpoint.
type FeatureExtraction = []float64

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
