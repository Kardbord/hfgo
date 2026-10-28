package hftypes

import (
	"maps"
	"slices"
)

// FillMaskRequest represents a fill mask
// inference request to the API for a single
// input.
type FillMaskRequest struct {
	// The input text with masked tokens.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for mask filling
	Parameters *FillMaskParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
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
func (r *FillMaskRequest) Clone() FillMaskRequest {
	if r == nil {
		return FillMaskRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*FillMaskParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

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
