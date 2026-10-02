package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ClassifyTokens sends a token classification request for a single input.
func ClassifyTokens(
	opts hfopts.Options,
	req hftypes.TokenClassificationRequest,
) ([]hftypes.TokenClassification, error) {
	return doJSONInference[hftypes.TokenClassificationRequest, []hftypes.TokenClassification](
		opts,
		providers.TaskTokenClassification,
		req,
	)
}

// ClassifyTokensBatch sends a token classification request for a batch of inputs.
func ClassifyTokensBatch(
	opts hfopts.Options,
	req hftypes.TokenClassificationBatchRequest,
) ([][]hftypes.TokenClassification, error) {
	return doJSONInference[hftypes.TokenClassificationBatchRequest, [][]hftypes.TokenClassification](
		opts,
		providers.TaskTokenClassification,
		req,
	)
}
