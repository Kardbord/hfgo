package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// Summarize sends a summarization request for a single input.
func Summarize(opts request.Options, req dto.SummarizationRequest) ([]dto.Summarization, error) {
	return doJSONInference[dto.SummarizationRequest, []dto.Summarization](
		opts,
		providers.TaskSummarization,
		req,
	)
}

// SummarizeBatch sends a summarization request for a batch of inputs.
func SummarizeBatch(
	opts request.Options,
	req dto.SummarizationBatchRequest,
) ([]dto.Summarization, error) {
	return doJSONInference[dto.SummarizationBatchRequest, []dto.Summarization](
		opts,
		providers.TaskSummarization,
		req,
	)
}
