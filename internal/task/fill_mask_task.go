package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// FillMask sends a fill mask request for a single input and returns the predictions.
func FillMask(
	opts hfopts.Options,
	req hftypes.FillMaskRequest,
) ([]hftypes.FillMaskPrediction, error) {
	return doJSONInference[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction](
		opts,
		providers.TaskFillMask,
		req,
	)
}

// FillMaskBatch sends a fill mask request for a batch of inputs and returns predictions per input.
func FillMaskBatch(
	opts hfopts.Options,
	req hftypes.FillMaskBatchRequest,
) ([][]hftypes.FillMaskPrediction, error) {
	return doJSONInference[hftypes.FillMaskBatchRequest, [][]hftypes.FillMaskPrediction](
		opts,
		providers.TaskFillMask,
		req,
	)
}
