package hftypes

import (
	"maps"
	"slices"
)

// ZeroShotTextClassificationRequest represents a zero-shot text
// classification request to the API for a single input.
type ZeroShotTextClassificationRequest struct {
	// The text to classify.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for zero-shot text classification.
	// Required.
	Parameters *ZeroShotTextClassificationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// ZeroShotTextClassificationParameters specify additional inference parameters
// for zero-shot text classification.
type ZeroShotTextClassificationParameters struct {
	// The set of possible class labels to classify the text into.
	// Required.
	CandidateLabels []string `json:"candidate_labels,omitempty"`

	// The sentence used in conjunction with candidate_labels to attempt
	// the text classification by replacing the placeholder with the
	// candidate labels.
	HypothesisTemplate *string `json:"hypothesis_template,omitempty"`

	// Whether multiple candidate labels can be true. If false, the scores
	// are normalized such that the sum of the label likelihoods for each
	// sequence is 1. If true, the labels are considered independent and
	// probabilities are normalized for each candidate.
	MultiLabel *bool `json:"multi_label,omitempty"`
}

// ZeroShotTextClassification represents a zero-shot text classification output.
type ZeroShotTextClassification struct {
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
func (r *ZeroShotTextClassificationRequest) Clone() ZeroShotTextClassificationRequest {
	if r == nil {
		return ZeroShotTextClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ZeroShotTextClassificationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *ZeroShotTextClassificationParameters) Clone() ZeroShotTextClassificationParameters {
	if p == nil {
		return ZeroShotTextClassificationParameters{}
	}
	out := *p
	out.CandidateLabels = slices.Clone(p.CandidateLabels)
	out.HypothesisTemplate = clonePtr(p.HypothesisTemplate)
	out.MultiLabel = clonePtr(p.MultiLabel)

	return out
}
