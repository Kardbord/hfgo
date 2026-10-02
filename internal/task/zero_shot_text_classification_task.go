package task

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ZeroShotClassifyText sends a zero-shot classification request for a single input.
func ZeroShotClassifyText(
	opts hfopts.Options,
	req hftypes.ZeroShotTextClassificationRequest,
) ([]hftypes.ZeroShotTextClassification, error) {
	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doJSONInference[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification](
		opts,
		providers.TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// ZeroShotClassifyTextBatch sends a zero-shot classification request for a batch of inputs
// and normalizes the batched response into a per-input structure.
func ZeroShotClassifyTextBatch(
	opts hfopts.Options,
	req hftypes.ZeroShotTextClassificationBatchRequest,
) ([][]hftypes.ZeroShotTextClassification, error) {
	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doJSONInference[hftypes.ZeroShotTextClassificationBatchRequest, []hftypes.ZeroShotTextClassificationBatched](
		opts,
		providers.TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return normalizeResponse(resp, req.Inputs)
}

func normalizeResponse(
	resp []hftypes.ZeroShotTextClassificationBatched,
	inputs []string,
) ([][]hftypes.ZeroShotTextClassification, error) {
	if len(resp) != len(inputs) {
		return nil, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindSerialization,
			Message: fmt.Sprintf(
				"response item count (%d) does not match input count (%d); API response format may have changed",
				len(resp),
				len(inputs),
			),
			Err: nil,
		}
	}

	result := make([][]hftypes.ZeroShotTextClassification, len(resp))

	for idx, item := range resp {
		if item.Sequence != inputs[idx] {
			return nil, &hferrors.SDKError{
				Kind: hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					`response item %d sequence does not match input; expected "%q" but got "%q"; API response format may have changed or order is not preserved`,
					idx,
					inputs[idx],
					item.Sequence,
				),
				Err: nil,
			}
		}

		if len(item.Labels) != len(item.Scores) {
			return nil, &hferrors.SDKError{
				Kind: hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					"response item %d has mismatched label and score counts (labels: %d, scores: %d); API response format may have changed",
					idx,
					len(item.Labels),
					len(item.Scores),
				),
				Err: nil,
			}
		}

		classifications := make([]hftypes.ZeroShotTextClassification, len(item.Labels))
		for j := range item.Labels {
			classifications[j] = hftypes.ZeroShotTextClassification{
				Label: item.Labels[j],
				Score: item.Scores[j],
			}
		}
		result[idx] = classifications
	}

	return result, nil
}
