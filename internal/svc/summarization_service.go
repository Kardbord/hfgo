package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// SummarizationService implements summarization calls.
type SummarizationService struct {
	opts request.Options
}

// NewSummarizationService builds a summarization service.
func NewSummarizationService(opts request.Options) SummarizationService {
	return SummarizationService{opts: opts}
}

// Summarize sends a summarization request for a single input.
func (s SummarizationService) Summarize(
	req dto.SummarizationRequest,
	opts ...request.Option,
) ([]dto.Summarization, error) {
	return doJSONInference[dto.SummarizationRequest, []dto.Summarization](
		s.opts.With(opts...),
		providers.TaskSummarization,
		req,
	)
}

// SummarizeBatch sends a summarization request for a batch of inputs.
func (s SummarizationService) SummarizeBatch(
	req dto.SummarizationBatchRequest,
	opts ...request.Option,
) ([]dto.Summarization, error) {
	return doJSONInference[dto.SummarizationBatchRequest, []dto.Summarization](
		s.opts.With(opts...),
		providers.TaskSummarization,
		req,
	)
}
