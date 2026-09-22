package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// summarizationService implements summarization calls using the configured request options.
type summarizationService struct {
	opts request.Options
}

// newSummarizationService builds a summarization service with a snapshot of the provided options.
func newSummarizationService(opts request.Options) summarizationService {
	return summarizationService{opts: opts}
}

// summarize sends a summarization request for a single input and returns the output.
func (s summarizationService) summarize(
	req SummarizationRequest,
	opts ...Option,
) ([]Summarization, error) {
	return doJSONInference[SummarizationRequest, []Summarization](
		s.opts.With(opts...),
		providers.TaskSummarization,
		req,
	)
}

// summarizeBatch sends a summarization request for a batch of inputs and returns a flat list of outputs.
func (s summarizationService) summarizeBatch(
	req SummarizationBatchRequest,
	opts ...Option,
) ([]Summarization, error) {
	return doJSONInference[SummarizationBatchRequest, []Summarization](
		s.opts.With(opts...),
		providers.TaskSummarization,
		req,
	)
}
