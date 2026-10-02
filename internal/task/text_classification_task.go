package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ClassifyText sends a text classification request for a single input and
// unwraps the outer API array to return the flat classification list.
func ClassifyText(
	opts hfopts.Options,
	req hftypes.TextClassificationRequest,
) ([]hftypes.TextClassification, error) {
	resp, err := doJSONInference[hftypes.TextClassificationRequest, [][]hftypes.TextClassification](
		opts,
		providers.TaskTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	if len(resp) < 1 {
		return nil, nil
	}

	return resp[0], nil
}

// ClassifyTextBatch sends a text classification request for a batch of inputs
// and returns classifications per input. When the API returns a flat list
// for batch inputs it is reshaped into a per-input structure.
func ClassifyTextBatch(
	opts hfopts.Options,
	req hftypes.TextClassificationBatchRequest,
) ([][]hftypes.TextClassification, error) {
	resp, err := doJSONInference[hftypes.TextClassificationBatchRequest, [][]hftypes.TextClassification](
		opts,
		providers.TaskTextClassification,
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

func normalizeTextClassificationResponse(
	resp [][]hftypes.TextClassification,
	numInputs int,
) [][]hftypes.TextClassification {
	if numInputs > 1 && len(resp) == 1 && len(resp[0]) == numInputs {
		reshaped := make([][]hftypes.TextClassification, numInputs)
		for i := range numInputs {
			reshaped[i] = []hftypes.TextClassification{resp[0][i]}
		}

		return reshaped
	}

	return resp
}
