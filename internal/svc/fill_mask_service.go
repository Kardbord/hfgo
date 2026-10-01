package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// FillMaskService implements fill mask calls using the configured request options.
type FillMaskService struct {
	opts request.Options
}

// NewFillMaskService builds a fill mask service with a snapshot of the provided options.
func NewFillMaskService(opts request.Options) FillMaskService {
	return FillMaskService{opts: opts}
}

// Fill sends a fill mask request for a single input and returns the predictions.
func (s FillMaskService) Fill(
	req dto.FillMaskRequest,
	opts ...request.Option,
) ([]dto.FillMaskPrediction, error) {
	return doJSONInference[dto.FillMaskRequest, []dto.FillMaskPrediction](
		s.opts.With(opts...),
		providers.TaskFillMask,
		req,
	)
}

// FillBatch sends a fill mask request for a batch of inputs and returns predictions per input.
func (s FillMaskService) FillBatch(
	req dto.FillMaskBatchRequest,
	opts ...request.Option,
) ([][]dto.FillMaskPrediction, error) {
	return doJSONInference[dto.FillMaskBatchRequest, [][]dto.FillMaskPrediction](
		s.opts.With(opts...),
		providers.TaskFillMask,
		req,
	)
}
