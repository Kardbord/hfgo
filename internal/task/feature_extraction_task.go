package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ExtractFeatures sends a feature extraction request for a single input.
func ExtractFeatures(
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtraction, error) {
	return doJSONInference[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction](
		opts,
		providers.TaskFeatureExtraction,
		req,
	)
}

// ExtractFeaturesBatch sends a feature extraction request for a batch of inputs.
func ExtractFeaturesBatch(
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtraction, error) {
	return doJSONInference[hftypes.FeatureExtractionBatchRequest, []hftypes.FeatureExtraction](
		opts,
		providers.TaskFeatureExtraction,
		req,
	)
}
