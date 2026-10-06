//nolint:dupl // Similar structure to other task functions by design
package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ClassifyTokens sends a token classification request for a single input.
func ClassifyTokens(
	opts hfopts.Options,
	req hftypes.TokenClassificationRequest,
) ([]hftypes.TokenClassification, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TokenClassificationProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TokenClassificationEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.TokenClassificationCodec(), req)
}

// ClassifyTokensBatch sends a token classification request for a batch of inputs.
func ClassifyTokensBatch(
	opts hfopts.Options,
	req hftypes.TokenClassificationBatchRequest,
) ([][]hftypes.TokenClassification, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TokenClassificationBatchProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TokenClassificationBatchEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.TokenClassificationBatchCodec(), req)
}
