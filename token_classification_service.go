package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// tokenClassificationService implements token classification calls using the configured request options.
type tokenClassificationService struct {
	opts request.Options
}

// newTokenClassificationService builds a token classification service with a snapshot of the provided options.
func newTokenClassificationService(opts request.Options) tokenClassificationService {
	return tokenClassificationService{opts: opts}
}

// classify sends a token classification request for a single input and returns the classified tokens.
func (s tokenClassificationService) classify(
	req TokenClassificationRequest,
	opts ...Option,
) ([]TokenClassification, error) {
	return doModelInference[TokenClassificationRequest, []TokenClassification](
		s.opts.With(opts...),
		TaskTokenClassification,
		req,
	)
}

// classifyBatch sends a token classification request for a batch of inputs and returns classified tokens per input.
func (s tokenClassificationService) classifyBatch(
	req TokenClassificationBatchRequest,
	opts ...Option,
) ([][]TokenClassification, error) {
	return doModelInference[TokenClassificationBatchRequest, [][]TokenClassification](
		s.opts.With(opts...),
		TaskTokenClassification,
		req,
	)
}
