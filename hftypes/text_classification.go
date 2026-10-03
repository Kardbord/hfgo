package hftypes

import "slices"

// TextClassificationRequest represents a text classification
// inference request to the API for a single input.
type TextClassificationRequest struct {
	Input      string                        `json:"inputs"`
	Parameters *TextClassificationParameters `json:"parameters,omitempty"`
}

// TextClassificationBatchRequest represents a batched text classification
// inference request to the API for multiple inputs.
type TextClassificationBatchRequest struct {
	Inputs     []string                      `json:"inputs"`
	Parameters *TextClassificationParameters `json:"parameters,omitempty"`
}

// TextClassificationParameters specify additional inference
// parameters for text classification.
type TextClassificationParameters struct {
	FunctionToApply *string `json:"function_to_apply,omitempty"`
	TopK            *int    `json:"top_k,omitempty"`
}

//nolint:revive // value constants, not configuration keys
const (
	TextClassificationFuncSigmoid = "sigmoid"
	TextClassificationFuncSoftmax = "softmax"
	TextClassificationFuncNone    = "none"
)

// TextClassification represents a text classification output.
type TextClassification struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// Clone returns a deep defensive copy of the request.
func (r *TextClassificationRequest) Clone() TextClassificationRequest {
	if r == nil {
		return TextClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TextClassificationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the batch request.
func (r *TextClassificationBatchRequest) Clone() TextClassificationBatchRequest {
	if r == nil {
		return TextClassificationBatchRequest{}
	}
	out := *r
	out.Inputs = slices.Clone(r.Inputs)
	out.Parameters = cloneStructPtr(r.Parameters, (*TextClassificationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *TextClassificationParameters) Clone() TextClassificationParameters {
	if p == nil {
		return TextClassificationParameters{}
	}
	out := *p
	out.FunctionToApply = clonePtr(p.FunctionToApply)
	out.TopK = clonePtr(p.TopK)

	return out
}
