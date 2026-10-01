package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// FillMask sends a fill mask request for a single input and returns the predictions.
func FillMask(opts request.Options, req dto.FillMaskRequest) ([]dto.FillMaskPrediction, error) {
	return doJSONInference[dto.FillMaskRequest, []dto.FillMaskPrediction](
		opts,
		providers.TaskFillMask,
		req,
	)
}

// FillMaskBatch sends a fill mask request for a batch of inputs and returns predictions per input.
func FillMaskBatch(
	opts request.Options,
	req dto.FillMaskBatchRequest,
) ([][]dto.FillMaskPrediction, error) {
	return doJSONInference[dto.FillMaskBatchRequest, [][]dto.FillMaskPrediction](
		opts,
		providers.TaskFillMask,
		req,
	)
}
