package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// TextClassificationService implements text classification calls.
type TextClassificationService struct {
	opts request.Options
}

// NewTextClassificationService builds a text classification service.
func NewTextClassificationService(opts request.Options) TextClassificationService {
	return TextClassificationService{opts: opts}
}

// Classify sends a text classification request for a single input and
// unwraps the outer API array to return the flat classification list.
func (s TextClassificationService) Classify(
	req dto.TextClassificationRequest,
	opts ...request.Option,
) ([]dto.TextClassification, error) {
	optsOverride := s.opts.With(opts...)

	resp, err := doJSONInference[dto.TextClassificationRequest, [][]dto.TextClassification](
		optsOverride,
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

// ClassifyBatch sends a text classification request for a batch of inputs
// and returns classifications per input. When the API returns a flat list
// for batch inputs it is reshaped into a per-input structure.
func (s TextClassificationService) ClassifyBatch(
	req dto.TextClassificationBatchRequest,
	opts ...request.Option,
) ([][]dto.TextClassification, error) {
	optsOverride := s.opts.With(opts...)

	resp, err := doJSONInference[dto.TextClassificationBatchRequest, [][]dto.TextClassification](
		optsOverride,
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
