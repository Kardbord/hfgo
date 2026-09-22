package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// fillMaskService implements fill mask calls using the configured request options.
type fillMaskService struct {
	opts request.Options
}

// newFillMaskService builds a fill mask service with a snapshot of the provided options.
func newFillMaskService(opts request.Options) fillMaskService {
	return fillMaskService{opts: opts}
}

// fill sends a fill mask request for a single input and returns the predictions.
func (s fillMaskService) fill(
	req FillMaskRequest,
	opts ...Option,
) ([]FillMaskPrediction, error) {
	return doJSONInference[FillMaskRequest, []FillMaskPrediction](
		s.opts.With(opts...),
		providers.TaskFillMask,
		req,
	)
}

// fillBatch sends a fill mask request for a batch of inputs and returns predictions per input.
func (s fillMaskService) fillBatch(
	req FillMaskBatchRequest,
	opts ...Option,
) ([][]FillMaskPrediction, error) {
	return doJSONInference[FillMaskBatchRequest, [][]FillMaskPrediction](
		s.opts.With(opts...),
		providers.TaskFillMask,
		req,
	)
}
