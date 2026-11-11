package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// RecognizeSpeech sends a speech recognition request.
func RecognizeSpeech(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.SpeechRecognitionRequest,
) (hftypes.SpeechRecognitionResponse, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.SpeechRecognitionProvider](opts.Provider)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	endpoint, err := prov.SpeechRecognitionEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.SpeechRecognitionCodec(), req)
}
