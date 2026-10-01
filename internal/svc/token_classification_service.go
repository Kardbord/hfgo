package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// TokenClassificationService implements token classification calls.
type TokenClassificationService struct {
	opts request.Options
}

// NewTokenClassificationService builds a token classification service.
func NewTokenClassificationService(opts request.Options) TokenClassificationService {
	return TokenClassificationService{opts: opts}
}

// Classify sends a token classification request for a single input.
func (s TokenClassificationService) Classify(
	req dto.TokenClassificationRequest,
	opts ...request.Option,
) ([]dto.TokenClassification, error) {
	return doJSONInference[dto.TokenClassificationRequest, []dto.TokenClassification](
		s.opts.With(opts...),
		providers.TaskTokenClassification,
		req,
	)
}

// ClassifyBatch sends a token classification request for a batch of inputs.
func (s TokenClassificationService) ClassifyBatch(
	req dto.TokenClassificationBatchRequest,
	opts ...request.Option,
) ([][]dto.TokenClassification, error) {
	return doJSONInference[dto.TokenClassificationBatchRequest, [][]dto.TokenClassification](
		s.opts.With(opts...),
		providers.TaskTokenClassification,
		req,
	)
}
