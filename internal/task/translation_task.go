package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// Translate sends a translation request for a single input.
func Translate(
	opts hfopts.Options,
	req hftypes.TranslationRequest,
) ([]hftypes.Translation, error) {
	return doJSONInference[hftypes.TranslationRequest, []hftypes.Translation](
		opts,
		hfproviders.TaskTranslation,
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
		hfproviders.TaskTranslation,
		req,
	)
}
