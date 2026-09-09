package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// textClassificationService implements text classification calls using the configured request options.
type textClassificationService struct {
	opts request.Options
}

// newTextClassificationService builds a text classification service with a snapshot of the provided options.
func newTextClassificationService(opts request.Options) textClassificationService {
	return textClassificationService{opts: opts}
}

// classify sends a text classification request for a single input and returns the classifications.
func (s textClassificationService) classify(
	req TextClassificationRequest,
	opts ...Option,
) ([]TextClassification, error) {
	optsOverride := s.opts.With(opts...)

	// NOTE: The API documentation indicates that this should return an JSON array of
	// TextClassification objects, but in reality it returns an array of arrays, where
	// the outer array contains only a single entry (the inner array), and the inner array
	// contains a list of TextClassification objects.
	resp, err := doModelInference[TextClassificationRequest, [][]TextClassification](
		optsOverride,
		TaskTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	// Unexpected, but technically legal API response.
	if len(resp) < 1 {
		return nil, nil
	}

	return resp[0], nil
}

// classifyBatch sends a text classification request for a batch of inputs and returns classifications per input.
func (s textClassificationService) classifyBatch(
	req TextClassificationBatchRequest,
	opts ...Option,
) ([][]TextClassification, error) {
	optsOverride := s.opts.With(opts...)

	resp, err := doModelInference[TextClassificationBatchRequest, [][]TextClassification](
		optsOverride,
		TaskTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	if req.Parameters != nil && req.Parameters.TopK != nil {
		return resp, nil
	}

	return normalizeTextClassificationResponse(resp, len(req.Inputs)), nil
}

// normalizeTextClassificationResponse handles a quirk in the HuggingFace API where
// the response format differs based on whether the TopK parameter is explicitly set:
//
//   - When TopK is unset (nil): Returns [[all classifications together]] (flat format)
//     and needs to be reshaped to [[classifications for input1], [classifications for input2], ...]
//
// This function detects the flat format case and reshapes it to the expected per-input
// format for API consistency. The detection heuristic is:
// - Single outer array (len(resp) == 1)
// - Number of inner classifications equals number of inputs (len(resp[0]) == numInputs)
// - More than one input was sent (numInputs > 1)
//
// When all conditions are met, we reshape [[class1, class2, class3]] into
// [[class1], [class2], [class3]] to maintain consistent per-input structure.
//
// NOTE: This function should only be called when TopK was not explicitly set.
// Calling it with TopK-set responses may cause data corruption.
func normalizeTextClassificationResponse(
	resp [][]TextClassification,
	numInputs int,
) [][]TextClassification {
	if numInputs > 1 && len(resp) == 1 && len(resp[0]) == numInputs {
		// Reshape flat format to per-input format
		reshaped := make([][]TextClassification, numInputs)
		for i := range numInputs {
			reshaped[i] = []TextClassification{resp[0][i]}
		}

		return reshaped
	}

	return resp
}
