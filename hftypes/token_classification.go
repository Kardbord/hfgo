package hftypes

import (
	"maps"
	"slices"
)

// TokenClassificationRequest represents a token classification
// inference request to the API for a single input.
type TokenClassificationRequest struct {
	// The text to classify tokens from.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for token classification.
	Parameters *TokenClassificationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// TokenClassificationParameters specify additional inference
// parameters for token classification.
type TokenClassificationParameters struct {
	// A list of labels to ignore in the classification results.
	IgnoreLabels []string `json:"ignore_labels,omitempty"`

	// The number of overlapping tokens between chunks when splitting the input text.
	Stride *int `json:"stride,omitempty"`

	// The strategy used to fuse tokens based on model predictions.
	// Possible values: "none", "simple", "first", "average", "max".
	AggregationStrategy *string `json:"aggregation_strategy,omitempty"`
}

const (
	// TokenClassificationAggregationNone does not aggregate tokens.
	// Each token is classified individually.
	TokenClassificationAggregationNone = "none"
	// TokenClassificationAggregationSimple groups consecutive tokens with the same
	// label into a single entity.
	TokenClassificationAggregationSimple = "simple"
	// TokenClassificationAggregationFirst is similar to "simple", but also preserves
	// word integrity using the label predicted for the first token in a word.
	TokenClassificationAggregationFirst = "first"
	// TokenClassificationAggregationAverage is similar to "simple", but also preserves
	// word integrity using the label with the highest score, averaged across the
	// word's tokens.
	TokenClassificationAggregationAverage = "average"
	// TokenClassificationAggregationMax is similar to "simple", but also preserves
	// word integrity using the label with the highest score across the word's tokens.
	TokenClassificationAggregationMax = "max"
)

// TokenClassification represents a token classification output.
type TokenClassification struct {
	// The predicted label for a group of one or more tokens.
	// Present when an aggregation strategy other than "none" is used.
	EntityGroup *string `json:"entity_group,omitempty"`

	// The predicted label for a single token.
	// Present when aggregation_strategy is "none".
	Entity *string `json:"entity,omitempty"`

	// The associated score / probability.
	Score float64 `json:"score"`

	// The corresponding text.
	Word string `json:"word"`

	// The character position in the input where this group begins.
	Start int `json:"start"`

	// The character position in the input where this group ends.
	End int `json:"end"`

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
func (r *TokenClassificationRequest) Clone() TokenClassificationRequest {
	if r == nil {
		return TokenClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TokenClassificationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *TokenClassificationParameters) Clone() TokenClassificationParameters {
	if p == nil {
		return TokenClassificationParameters{}
	}
	out := *p
	out.IgnoreLabels = slices.Clone(p.IgnoreLabels)
	out.Stride = clonePtr(p.Stride)
	out.AggregationStrategy = clonePtr(p.AggregationStrategy)

	return out
}
