package dto

import "slices"

// TokenClassificationRequest represents a token classification
// inference request to the API for a single input.
type TokenClassificationRequest struct {
	// The text to classify tokens from.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for token classification.
	Parameters *TokenClassificationParameters `json:"parameters,omitempty"`
}

// TokenClassificationBatchRequest represents a batched token classification
// request to the API for multiple inputs.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
type TokenClassificationBatchRequest struct {
	// The texts to classify tokens from.
	// Required.
	Inputs []string `json:"inputs"`

	// Additional inference parameters for token classification.
	Parameters *TokenClassificationParameters `json:"parameters,omitempty"`
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
}

// Clone returns a deep defensive copy of the request.
func (r *TokenClassificationRequest) Clone() TokenClassificationRequest {
	if r == nil {
		return TokenClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TokenClassificationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the request.
func (r *TokenClassificationBatchRequest) Clone() TokenClassificationBatchRequest {
	if r == nil {
		return TokenClassificationBatchRequest{}
	}
	out := *r
	out.Inputs = slices.Clone(r.Inputs)
	out.Parameters = cloneStructPtr(r.Parameters, (*TokenClassificationParameters).Clone)

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
