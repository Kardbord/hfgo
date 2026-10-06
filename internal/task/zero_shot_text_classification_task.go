package task

import (
	"fmt"

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

// ZeroShotClassifyTextBatch sends a zero-shot classification request for a batch of inputs
// and normalizes the batched response into a per-input structure.
func ZeroShotClassifyTextBatch(
	opts hfopts.Options,
	req hftypes.ZeroShotTextClassificationBatchRequest,
) ([][]hftypes.ZeroShotTextClassification, error) {
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

	prov, err := hfproviders.AsProvider[hfproviders.ZeroShotTextClassificationBatchProvider](
		opts.Provider,
	)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.ZeroShotTextClassificationBatchEndpoint(hfproviders.EndpointParams{
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

	resp, err := doInference(opts, endpoint, prov.ZeroShotTextClassificationBatchCodec(), req)
	if err != nil {
		return nil, err
	}

	return normalizeResponse(resp, req.Inputs)
}

func normalizeResponse(
	resp []hftypes.ZeroShotTextClassificationBatched,
	inputs []string,
) ([][]hftypes.ZeroShotTextClassification, error) {
	if len(resp) != len(inputs) {
		return nil, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindSerialization,
			Message: fmt.Sprintf(
				"response item count (%d) does not match input count (%d); API response format may have changed",
				len(resp),
				len(inputs),
			),
			Err: nil,
		}
	}

	result := make([][]hftypes.ZeroShotTextClassification, len(resp))

	for idx, item := range resp {
		if item.Sequence != inputs[idx] {
			return nil, &hferrors.SDKError{
				Kind: hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					`response item %d sequence does not match input; expected "%q" but got "%q"; API response format may have changed or order is not preserved`,
					idx,
					inputs[idx],
					item.Sequence,
				),
				Err: nil,
			}
		}

		if len(item.Labels) != len(item.Scores) {
			return nil, &hferrors.SDKError{
				Kind: hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf(
					"response item %d has mismatched label and score counts (labels: %d, scores: %d); API response format may have changed",
					idx,
					len(item.Labels),
					len(item.Scores),
				),
				Err: nil,
			}
		}

		classifications := make([]hftypes.ZeroShotTextClassification, len(item.Labels))
		for j := range item.Labels {
			classifications[j] = hftypes.ZeroShotTextClassification{
				Label: item.Labels[j],
				Score: item.Scores[j],
			}
		}
		result[idx] = classifications
	}

	return result, nil
}
