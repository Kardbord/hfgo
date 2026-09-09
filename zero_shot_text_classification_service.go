package hfgo

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/internal/request"
)

// zeroShotTextClassificationService implements zero-shot text classification
// calls using the configured request options.
type zeroShotTextClassificationService struct {
	opts request.Options
}

// newZeroShotTextClassificationService builds a zero-shot text classification service
// with a snapshot of the provided options.
func newZeroShotTextClassificationService(opts request.Options) zeroShotTextClassificationService {
	return zeroShotTextClassificationService{opts: opts}
}

// classify sends a zero-shot text classification request for a single input and returns the classifications.
func (s zeroShotTextClassificationService) classify(
	req ZeroShotTextClassificationRequest,
	opts ...Option,
) ([]ZeroShotTextClassification, error) {
	optsOverride := s.opts.With(opts...)

	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doModelInference[ZeroShotTextClassificationRequest, []ZeroShotTextClassification](
		optsOverride,
		TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// classifyBatch sends a zero-shot text classification request for a batch of inputs
// and returns classifications for each input.
func (s zeroShotTextClassificationService) classifyBatch(
	req ZeroShotTextClassificationBatchRequest,
	opts ...Option,
) ([][]ZeroShotTextClassification, error) {
	optsOverride := s.opts.With(opts...)

	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doModelInference[ZeroShotTextClassificationBatchRequest, []zeroShotTextClassificationBatched](
		optsOverride,
		TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return normalizeZeroShotTextClassificationResponse(resp, req.Inputs)
}

// normalizeZeroShotTextClassificationResponse handles a quirk in the HuggingFace API where
// the response format is a list of labels and a list of scores.
//
//	[
//		{
//			"sequence:" "sequence 1",
//			"labels": ["label1", "label2"],
//			"scores": [score1, score2]
//		},
//		{
//			"sequence:" "sequence 2",
//			"labels": ["label1", "label2"],
//			"scores": [score1, score2]
//		}
//	]
//
// Because batched request handling is available but undocumented, we're taking the liberty for
// now of normalizing the response into the same format as text classification for consistency.
//
//	[ // This list is ordered the same as the input list
//		[
//			{"label": "label1", "score": score1},
//			{"label": "label2", "score": score2}
//		],
//		[
//			{"label": "label1", "score": score1},
//			{"label": "label2", "score": score2}
//		],
//	]
//
// NOTE: Should batched zero-shot classification become officially supported and documented,
// we'll want to simply return the same format as the upstream API.
func normalizeZeroShotTextClassificationResponse(
	resp []zeroShotTextClassificationBatched,
	inputs []string,
) ([][]ZeroShotTextClassification, error) {
	// Validate response length matches input length
	if len(resp) != len(inputs) {
		return nil, &SDKError{
			Kind: SDKErrorKindSerialization,
			Message: fmt.Sprintf(
				"response item count (%d) does not match input count (%d); API response format may have changed",
				len(resp),
				len(inputs),
			),
			Err: nil,
		}
	}

	result := make([][]ZeroShotTextClassification, len(resp))

	//nolint:varnamelen // "i" is commonly used as an outer loop index variable
	for i, item := range resp {
		// Validate that the response sequence matches the input
		if item.Sequence != inputs[i] {
			return nil, &SDKError{
				Kind: SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					`response item %d sequence does not match input; expected "%q" but got "%q"; API response format may have changed or order is not preserved`,
					i,
					inputs[i],
					item.Sequence,
				),
				Err: nil,
			}
		}

		// Validate that labels and scores have matching lengths
		if len(item.Labels) != len(item.Scores) {
			return nil, &SDKError{
				Kind: SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					"response item %d has mismatched label and score counts (labels: %d, scores: %d); API response format may have changed",
					i,
					len(item.Labels),
					len(item.Scores),
				),
				Err: nil,
			}
		}

		classifications := make([]ZeroShotTextClassification, len(item.Labels))
		for j := range item.Labels {
			classifications[j] = ZeroShotTextClassification{
				Label: item.Labels[j],
				Score: item.Scores[j],
			}
		}
		result[i] = classifications
	}

	return result, nil
}
