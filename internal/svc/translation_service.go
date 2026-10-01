package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// TranslationService implements translation calls.
type TranslationService struct {
	opts request.Options
}

// NewTranslationService builds a translation service.
func NewTranslationService(opts request.Options) TranslationService {
	return TranslationService{opts: opts}
}

// Translate sends a translation request for a single input.
func (s TranslationService) Translate(
	req dto.TranslationRequest,
	opts ...request.Option,
) ([]dto.Translation, error) {
	return doJSONInference[dto.TranslationRequest, []dto.Translation](
		s.opts.With(opts...),
		providers.TaskTranslation,
		req,
	)
}

// TranslateBatch sends a translation request for a batch of inputs.
func (s TranslationService) TranslateBatch(
	req dto.TranslationBatchRequest,
	opts ...request.Option,
) ([]dto.Translation, error) {
	return doJSONInference[dto.TranslationBatchRequest, []dto.Translation](
		s.opts.With(opts...),
		providers.TaskTranslation,
		req,
	)
}
