package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// GenerateVideo sends a text-to-video request and returns the generated video
// along with the media type the API advertised for it.
func GenerateVideo(
	opts hfopts.Options,
	req hftypes.TextToVideoRequest,
) (hftypes.TextToVideoResponse, error) {
	if err := validateDispatch(opts); err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextToVideoProvider](opts.Provider)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	endpoint, err := prov.TextToVideoEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return hftypes.TextToVideoResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.TextToVideoCodec(), req)
}
