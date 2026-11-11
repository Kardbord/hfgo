package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// GenerateImage sends a text-to-image request and returns the generated image
// along with the media type the API advertised for it.
func GenerateImage(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.TextToImageRequest,
) (hftypes.TextToImageResponse, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextToImageProvider](opts.Provider)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	endpoint, err := prov.TextToImageEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return hftypes.TextToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.TextToImageCodec(), req)
}
