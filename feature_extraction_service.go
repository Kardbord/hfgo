package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// featureExtractionService implements feature extraction calls using the configured request options.
type featureExtractionService struct {
	opts request.Options
}

// newFeatureExtractionService builds a feature extraction service with a snapshot of the provided options.
func newFeatureExtractionService(opts request.Options) featureExtractionService {
	return featureExtractionService{opts: opts}
}

// extract sends a feature extraction request for a single input and returns the embedding vector.
func (s featureExtractionService) extract(
	req FeatureExtractionRequest,
	opts ...Option,
) (FeatureExtraction, error) {
	return doModelInference[FeatureExtractionRequest, FeatureExtraction](
		s.opts.With(opts...),
		TaskFeatureExtraction,
		req,
	)
}

// extractBatch sends a feature extraction request for a batch of inputs and returns embedding vectors per input.
func (s featureExtractionService) extractBatch(
	req FeatureExtractionBatchRequest,
	opts ...Option,
) ([]FeatureExtraction, error) {
	return doModelInference[FeatureExtractionBatchRequest, []FeatureExtraction](
		s.opts.With(opts...),
		TaskFeatureExtraction,
		req,
	)
}
