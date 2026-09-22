package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// translationService implements translation calls using the configured request options.
type translationService struct {
	opts request.Options
}

// newTranslationService builds a translation service with a snapshot of the provided options.
func newTranslationService(opts request.Options) translationService {
	return translationService{opts: opts}
}

// translate sends a translation request for a single input and returns the output.
func (s translationService) translate(
	req TranslationRequest,
	opts ...Option,
) ([]Translation, error) {
	return doJSONInference[TranslationRequest, []Translation](
		s.opts.With(opts...),
		providers.TaskTranslation,
		req,
	)
}

// translateBatch sends a translation request for a batch of inputs and returns a flat list of outputs.
func (s translationService) translateBatch(
	req TranslationBatchRequest,
	opts ...Option,
) ([]Translation, error) {
	return doJSONInference[TranslationBatchRequest, []Translation](
		s.opts.With(opts...),
		providers.TaskTranslation,
		req,
	)
}
