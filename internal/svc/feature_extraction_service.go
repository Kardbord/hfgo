package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// FeatureExtractionService implements feature extraction calls.
type FeatureExtractionService struct {
	opts request.Options
}

// NewFeatureExtractionService builds a feature extraction service.
func NewFeatureExtractionService(opts request.Options) FeatureExtractionService {
	return FeatureExtractionService{opts: opts}
}

// Extract sends a feature extraction request for a single input.
func (s FeatureExtractionService) Extract(
	req dto.FeatureExtractionRequest,
	opts ...request.Option,
) (dto.FeatureExtraction, error) {
	return doJSONInference[dto.FeatureExtractionRequest, dto.FeatureExtraction](
		s.opts.With(opts...),
		providers.TaskFeatureExtraction,
		req,
	)
}

// ExtractBatch sends a feature extraction request for a batch of inputs.
func (s FeatureExtractionService) ExtractBatch(
	req dto.FeatureExtractionBatchRequest,
	opts ...request.Option,
) ([]dto.FeatureExtraction, error) {
	return doJSONInference[dto.FeatureExtractionBatchRequest, []dto.FeatureExtraction](
		s.opts.With(opts...),
		providers.TaskFeatureExtraction,
		req,
	)
}
