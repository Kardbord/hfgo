package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// ClassifyText sends a text classification request for a single input.
func ClassifyText(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.TextClassificationRequest,
) ([]hftypes.TextClassification, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextClassificationProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TextClassificationEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.TextClassificationCodec(), req)
}
