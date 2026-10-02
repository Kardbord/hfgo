package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ExtractFeatures sends a feature extraction request for a single input.
func ExtractFeatures(
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtraction, error) {
	return doJSONInference[hftypes.FeatureExtractionRequest, hftypes.FeatureExtraction](
		opts,
		hfproviders.TaskFeatureExtraction,
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
		hfproviders.TaskFeatureExtraction,
		req,
	)
}
