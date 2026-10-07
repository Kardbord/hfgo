//nolint:dupl // Similar structure to other task functions by design
package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// Summarize sends a summarization request for a single input.
func Summarize(
	opts hfopts.Options,
	req hftypes.SummarizationRequest,
) ([]hftypes.Summarization, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.SummarizationProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.SummarizationEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.SummarizationCodec(), req)
}

// SummarizeBatch sends a summarization request for a batch of inputs.
func SummarizeBatch(
	opts hfopts.Options,
	req hftypes.SummarizationBatchRequest,
) ([]hftypes.Summarization, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.SummarizationBatchProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.SummarizationBatchEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.SummarizationBatchCodec(), req)
}
