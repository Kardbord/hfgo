package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// RecognizeSpeech sends a speech recognition request.
func RecognizeSpeech(
	opts hfopts.Options,
	req hftypes.SpeechRecognitionRequest,
) (hftypes.SpeechRecognitionResponse, error) {
	if err := validateDispatch(opts); err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.SpeechRecognitionProvider](opts.Provider)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	endpoint, err := prov.SpeechRecognitionEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.SpeechRecognitionCodec(), req)
}
