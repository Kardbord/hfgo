package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ClassifyTokens sends a token classification request for a single input.
func ClassifyTokens(
	opts request.Options,
	req dto.TokenClassificationRequest,
) ([]dto.TokenClassification, error) {
	return doJSONInference[dto.TokenClassificationRequest, []dto.TokenClassification](
		opts,
		providers.TaskTokenClassification,
		req,
	)
}

// ClassifyTokensBatch sends a token classification request for a batch of inputs.
func ClassifyTokensBatch(
	opts request.Options,
	req dto.TokenClassificationBatchRequest,
) ([][]dto.TokenClassification, error) {
	return doJSONInference[dto.TokenClassificationBatchRequest, [][]dto.TokenClassification](
		opts,
		providers.TaskTokenClassification,
		req,
	)
}
