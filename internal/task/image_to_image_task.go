package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// GenerateImageToImage sends an image-to-image request and returns the
// generated image along with the media type the API advertised for it.
func GenerateImageToImage(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.ImageToImageRequest,
) (hftypes.ImageToImageResponse, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.ImageToImageProvider](opts.Provider)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	endpoint, err := prov.ImageToImageEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return hftypes.ImageToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.ImageToImageCodec(), req)
}
