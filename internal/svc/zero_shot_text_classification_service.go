package svc

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// ZeroShotTextClassificationService implements zero-shot text classification calls.
type ZeroShotTextClassificationService struct {
	opts request.Options
}

// NewZeroShotTextClassificationService builds a zero-shot classification service.
func NewZeroShotTextClassificationService(opts request.Options) ZeroShotTextClassificationService {
	return ZeroShotTextClassificationService{opts: opts}
}

// Classify sends a zero-shot classification request for a single input.
func (s ZeroShotTextClassificationService) Classify(
	req dto.ZeroShotTextClassificationRequest,
	opts ...request.Option,
) ([]dto.ZeroShotTextClassification, error) {
	optsOverride := s.opts.With(opts...)

	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doJSONInference[dto.ZeroShotTextClassificationRequest, []dto.ZeroShotTextClassification](
		optsOverride,
		providers.TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// ClassifyBatch sends a zero-shot classification request for a batch of inputs
// and normalizes the batched response into a per-input structure.
func (s ZeroShotTextClassificationService) ClassifyBatch(
	req dto.ZeroShotTextClassificationBatchRequest,
	opts ...request.Option,
) ([][]dto.ZeroShotTextClassification, error) {
	optsOverride := s.opts.With(opts...)

	if req.Parameters == nil || len(req.Parameters.CandidateLabels) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "candidate labels must be provided for zero-shot text classification",
			Err:     nil,
		}
	}

	resp, err := doJSONInference[dto.ZeroShotTextClassificationBatchRequest, []dto.ZeroShotTextClassificationBatched](
		optsOverride,
		providers.TaskZeroShotTextClassification,
		req,
	)
	if err != nil {
		return nil, err
	}

	return normalizeResponse(resp, req.Inputs)
}

func normalizeResponse(
	resp []dto.ZeroShotTextClassificationBatched,
	inputs []string,
) ([][]dto.ZeroShotTextClassification, error) {
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

	result := make([][]dto.ZeroShotTextClassification, len(resp))

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

		classifications := make([]dto.ZeroShotTextClassification, len(item.Labels))
		for j := range item.Labels {
			classifications[j] = dto.ZeroShotTextClassification{
				Label: item.Labels[j],
				Score: item.Scores[j],
			}
		}
		result[idx] = classifications
	}

	return result, nil
}
