package hftypes

import "slices"

// FillMaskRequest represents a fill mask
// inference request to the API for a single
// input.
type FillMaskRequest struct {
	// The input text with masked tokens.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for mask filling
	Parameters *FillMaskParameters `json:"parameters,omitempty"`
}

// FillMaskBatchRequest represents a batched fill mask
// request to the API for multiple masked inputs.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
type FillMaskBatchRequest struct {
	// The inputs with masked tokens.
	// Required.
	Inputs []string `json:"inputs"`

	// Additional inference parameters for mask filling
	Parameters *FillMaskParameters `json:"parameters,omitempty"`
}

// FillMaskParameters specify additional inference
// parameters for mask filling tasks.
type FillMaskParameters struct {
	// When passed, the model will limit the scores to the passed targets instead of looking up
	// in the whole vocabulary. If the provided targets are not in the model vocab, they will be
	// tokenized and the first resulting token will be used (with a warning, and that might be
	// slower).
	Targets []string `json:"targets,omitempty"`

	// When passed, overrides the number of predictions to return.
	TopK *int `json:"top_k,omitempty"`
}

// FillMaskPrediction represents a mask-filling output.
type FillMaskPrediction struct {
	// The input filled with the mask token prediction
	Sequence string `json:"sequence"`

	// The probability of the token prediction
	Score float64 `json:"score"`

	// The predicted token id (to replace the masked one).
	Token int `json:"token"`

	// The predicted token (to replace the masked one).
	TokenStr *string `json:"token_str"`
}

// Clone returns a deep defensive copy of the request.
func (r *FillMaskRequest) Clone() FillMaskRequest {
	if r == nil {
		return FillMaskRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*FillMaskParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the request.
func (r *FillMaskBatchRequest) Clone() FillMaskBatchRequest {
	if r == nil {
		return FillMaskBatchRequest{}
	}
	out := *r
	out.Inputs = slices.Clone(r.Inputs)
	out.Parameters = cloneStructPtr(r.Parameters, (*FillMaskParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *FillMaskParameters) Clone() FillMaskParameters {
	if p == nil {
		return FillMaskParameters{}
	}
	out := *p
	out.Targets = slices.Clone(p.Targets)
	out.TopK = clonePtr(p.TopK)

	return out
}
