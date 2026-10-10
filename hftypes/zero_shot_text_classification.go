package hftypes

import "slices"

// ZeroShotTextClassificationRequest represents a zero-shot text
// classification request to the API for a single input.
type ZeroShotTextClassificationRequest struct {
	// The text to classify.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for zero-shot text classification.
	// Required.
	Parameters *ZeroShotTextClassificationParameters `json:"parameters,omitempty"`
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
}

// Clone returns a deep defensive copy of the request.
func (r *ZeroShotTextClassificationRequest) Clone() ZeroShotTextClassificationRequest {
	if r == nil {
		return ZeroShotTextClassificationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ZeroShotTextClassificationParameters).Clone)

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
