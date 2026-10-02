package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// Summarize sends a summarization request for a single input.
func Summarize(
	opts hfopts.Options,
	req hftypes.SummarizationRequest,
) ([]hftypes.Summarization, error) {
	return doJSONInference[hftypes.SummarizationRequest, []hftypes.Summarization](
		opts,
		providers.TaskSummarization,
		req,
	)
}

// SummarizeBatch sends a summarization request for a batch of inputs.
func SummarizeBatch(
	opts hfopts.Options,
	req hftypes.SummarizationBatchRequest,
) ([]hftypes.Summarization, error) {
	return doJSONInference[hftypes.SummarizationBatchRequest, []hftypes.Summarization](
		opts,
		providers.TaskSummarization,
		req,
	)
}
