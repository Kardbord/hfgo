package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ExtractFeatures sends a feature extraction request for a single input.
func ExtractFeatures(
	opts request.Options,
	req dto.FeatureExtractionRequest,
) (dto.FeatureExtraction, error) {
	return doJSONInference[dto.FeatureExtractionRequest, dto.FeatureExtraction](
		opts,
		providers.TaskFeatureExtraction,
		req,
	)
}

// ExtractFeaturesBatch sends a feature extraction request for a batch of inputs.
func ExtractFeaturesBatch(
	opts request.Options,
	req dto.FeatureExtractionBatchRequest,
) ([]dto.FeatureExtraction, error) {
	return doJSONInference[dto.FeatureExtractionBatchRequest, []dto.FeatureExtraction](
		opts,
		providers.TaskFeatureExtraction,
		req,
	)
}
