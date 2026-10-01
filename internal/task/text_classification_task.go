package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ClassifyText sends a text classification request for a single input and
// unwraps the outer API array to return the flat classification list.
func ClassifyText(
	opts request.Options,
	req dto.TextClassificationRequest,
) ([]dto.TextClassification, error) {
	resp, err := doJSONInference[dto.TextClassificationRequest, [][]dto.TextClassification](
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
	opts request.Options,
	req dto.TextClassificationBatchRequest,
) ([][]dto.TextClassification, error) {
	resp, err := doJSONInference[dto.TextClassificationBatchRequest, [][]dto.TextClassification](
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
	resp [][]dto.TextClassification,
	numInputs int,
) [][]dto.TextClassification {
	if numInputs > 1 && len(resp) == 1 && len(resp[0]) == numInputs {
		reshaped := make([][]dto.TextClassification, numInputs)
		for i := range numInputs {
			reshaped[i] = []dto.TextClassification{resp[0][i]}
		}

		return reshaped
	}

	return resp
}
