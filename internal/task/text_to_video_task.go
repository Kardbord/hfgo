package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// GenerateVideo sends a text-to-video request and returns the generated video
// along with the media type the API advertised for it.
func GenerateVideo(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.TextToVideoRequest,
) (hftypes.TextToVideoResponse, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextToVideoProvider](opts.Provider)
	if err != nil {
		return hftypes.TextToVideoResponse{}, err
	}

	endpoint, err := prov.TextToVideoEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return hftypes.TextToVideoResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.TextToVideoCodec(), req)
}
