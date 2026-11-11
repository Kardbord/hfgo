package hftypes //nolint:dupl // Task DTOs are intentionally parallel; the shapes mirror the upstream schema.

import "maps"

// ImageClassificationRequest represents an image classification
// inference request to the API for a single input.
type ImageClassificationRequest struct {
	// The input image data as a base64-encoded string.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for image classification.
	// Optional.
	Parameters *ImageClassificationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
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

	// ProviderMetadata is an advanced escape hatch for provider-specific
	// response metadata not covered by the typed fields. Providers that
	// support it populate the map with additional upstream response data;
	// it stays nil otherwise. Generic JSON decoding does not populate it:
	// only provider-specific codecs do.
	ProviderMetadata map[string]any `json:"-"`
}

// Clone returns a deep defensive copy of the request. The ProviderParams map
// is copied as a new map, but its values are shared because their types are
// not known statically.
func (r *ImageClassificationRequest) Clone() ImageClassificationRequest {
	if r == nil {
		return ImageClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ImageClassificationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

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
