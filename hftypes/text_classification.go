package hftypes //nolint:dupl // Task DTOs are intentionally parallel; the shapes mirror the upstream schema.

import "maps"

// TextClassificationRequest represents a text classification
// inference request to the API for a single input.
type TextClassificationRequest struct {
	Input      string                        `json:"inputs"`
	Parameters *TextClassificationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
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
func (r *TextClassificationRequest) Clone() TextClassificationRequest {
	if r == nil {
		return TextClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TextClassificationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

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
