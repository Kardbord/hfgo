package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// Translate sends a translation request for a single input.
func Translate(
	opts hfopts.Options,
	req hftypes.TranslationRequest,
) ([]hftypes.Translation, error) {
	return doJSONInference[hftypes.TranslationRequest, []hftypes.Translation](
		opts,
		providers.TaskTranslation,
		req,
	)
}

// TranslateBatch sends a translation request for a batch of inputs.
func TranslateBatch(
	opts hfopts.Options,
	req hftypes.TranslationBatchRequest,
) ([]hftypes.Translation, error) {
	return doJSONInference[hftypes.TranslationBatchRequest, []hftypes.Translation](
		opts,
		providers.TaskTranslation,
		req,
	)
}
