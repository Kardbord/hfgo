//nolint:dupl // Similar structure to other task functions by design
package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// FillMask sends a fill mask request for a single input and returns the predictions.
func FillMask(
	opts hfopts.Options,
	req hftypes.FillMaskRequest,
) ([]hftypes.FillMaskPrediction, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FillMaskProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FillMaskEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.FillMaskCodec(), req)
}

// FillMaskBatch sends a fill mask request for a batch of inputs and returns predictions per input.
func FillMaskBatch(
	opts hfopts.Options,
	req hftypes.FillMaskBatchRequest,
) ([][]hftypes.FillMaskPrediction, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FillMaskBatchProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FillMaskBatchEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.FillMaskBatchCodec(), req)
}
