package hftypes

// ImageClassificationRequest represents an image classification
// inference request to the API for a single input.
type ImageClassificationRequest struct {
	// The input image data as a base64-encoded string.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for image classification.
	// Optional.
	Parameters *ImageClassificationParameters `json:"parameters,omitempty"`
}

// ImageClassificationParameters specify additional inference parameters
// for image classification tasks.
type ImageClassificationParameters struct {
	// The function to apply to the model outputs in order to
	// retrieve the scores.
	// Possible values: "sigmoid", "softmax", "none".
	FunctionToApply *string `json:"function_to_apply,omitempty"`

	// When specified, limits the output to the top K most probable classes.
	TopK *int `json:"top_k,omitempty"`
}

const (
	// ImageClassificationFuncSigmoid applies the sigmoid function to model outputs.
	ImageClassificationFuncSigmoid = "sigmoid"
	// ImageClassificationFuncSoftmax applies the softmax function to model outputs.
	ImageClassificationFuncSoftmax = "softmax"
	// ImageClassificationFuncNone applies no function to model outputs.
	ImageClassificationFuncNone = "none"
)

// ImageClassification represents an image classification task output.
type ImageClassification struct {
	// The predicted class label.
	Label string `json:"label"`

	// The corresponding probability.
	Score float64 `json:"score"`
}

// Clone returns a deep defensive copy of the request.
func (r *ImageClassificationRequest) Clone() ImageClassificationRequest {
	if r == nil {
		return ImageClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ImageClassificationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *ImageClassificationParameters) Clone() ImageClassificationParameters {
	if p == nil {
		return ImageClassificationParameters{}
	}
	out := *p
	out.FunctionToApply = clonePtr(p.FunctionToApply)
	out.TopK = clonePtr(p.TopK)

	return out
}
