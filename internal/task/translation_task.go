package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// Translate sends a translation request for a single input.
func Translate(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.TranslationRequest,
) ([]hftypes.Translation, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TranslationProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TranslationEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.TranslationCodec(), req)
}
