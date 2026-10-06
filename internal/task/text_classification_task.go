package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ClassifyText sends a text classification request for a single input.
func ClassifyText(
	opts hfopts.Options,
	req hftypes.TextClassificationRequest,
) ([]hftypes.TextClassification, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextClassificationProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TextClassificationEndpoint(hfproviders.EndpointParams{
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

	return doInference(opts, endpoint, prov.TextClassificationCodec(), req)
}

// ClassifyTextBatch sends a text classification request for a batch of inputs
// and returns classifications per input. When the API returns a flat list
// for batch inputs it is reshaped into a per-input structure.
func ClassifyTextBatch(
	opts hfopts.Options,
	req hftypes.TextClassificationBatchRequest,
) ([][]hftypes.TextClassification, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TextClassificationBatchProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.TextClassificationBatchEndpoint(hfproviders.EndpointParams{
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

	resp, err := doInference(opts, endpoint, prov.TextClassificationBatchCodec(), req)
	if err != nil {
		return nil, err
	}

	if req.Parameters != nil && req.Parameters.TopK != nil {
		return resp, nil
	}

	return normalizeTextClassificationResponse(resp, len(req.Inputs)), nil
}

func normalizeTextClassificationResponse(
	resp [][]hftypes.TextClassification,
	numInputs int,
) [][]hftypes.TextClassification {
	if numInputs > 1 && len(resp) == 1 && len(resp[0]) == numInputs {
		reshaped := make([][]hftypes.TextClassification, numInputs)
		for i := range numInputs {
			reshaped[i] = []hftypes.TextClassification{resp[0][i]}
		}

		return reshaped
	}

	return resp
}
