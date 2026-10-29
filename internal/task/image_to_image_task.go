package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// GenerateImageToImage sends an image-to-image request and returns the
// generated image along with the media type the API advertised for it.
func GenerateImageToImage(
	opts hfopts.Options,
	req hftypes.ImageToImageRequest,
) (hftypes.ImageToImageResponse, error) {
	if err := validateDispatch(opts); err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.ImageToImageProvider](opts.Provider)
	if err != nil {
		return hftypes.ImageToImageResponse{}, err
	}

	endpoint, err := prov.ImageToImageEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return hftypes.ImageToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.ImageToImageCodec(), req)
}
