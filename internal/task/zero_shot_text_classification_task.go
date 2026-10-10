package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ZeroShotClassifyText sends a zero-shot classification request for a single input.
func ZeroShotClassifyText(
	opts hfopts.Options,
	req hftypes.ZeroShotTextClassificationRequest,
) ([]hftypes.ZeroShotTextClassification, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	prov, err := hfproviders.AsProvider[hfproviders.ZeroShotTextClassificationProvider](
		opts.Provider,
	)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.ZeroShotTextClassificationEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.ZeroShotTextClassificationCodec(), req)
}
