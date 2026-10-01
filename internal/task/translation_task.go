package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// Translate sends a translation request for a single input.
func Translate(opts request.Options, req dto.TranslationRequest) ([]dto.Translation, error) {
	return doJSONInference[dto.TranslationRequest, []dto.Translation](
		opts,
		providers.TaskTranslation,
		req,
	)
}

// TranslateBatch sends a translation request for a batch of inputs.
func TranslateBatch(
	opts request.Options,
	req dto.TranslationBatchRequest,
) ([]dto.Translation, error) {
	return doJSONInference[dto.TranslationBatchRequest, []dto.Translation](
		opts,
		providers.TaskTranslation,
		req,
	)
}
