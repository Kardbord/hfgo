package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// GenerateImage sends a text-to-image request and returns the generated image
// along with the media type the API advertised for it.
func GenerateImage(
	opts hfopts.Options,
	req hftypes.TextToImageRequest,
) (hftypes.TextToImageResponse, error) {
	if err := validateDispatch(opts); err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextToImageProvider](opts.Provider)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}

	endpoint, err := prov.TextToImageEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return hftypes.TextToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.TextToImageCodec(), req)
}
