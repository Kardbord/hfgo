package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// QuestionAnsweringService implements question answering calls.
type QuestionAnsweringService struct {
	opts request.Options
}

// NewQuestionAnsweringService builds a question answering service.
func NewQuestionAnsweringService(opts request.Options) QuestionAnsweringService {
	return QuestionAnsweringService{opts: opts}
}

// Answer sends a question answering request and returns the answers.
// When TopK > 1 it performs a multi-answer inference; otherwise it wraps
// the single-answer response into a one-element list.
func (s QuestionAnsweringService) Answer(
	req dto.QuestionAnsweringRequest,
	opts ...request.Option,
) ([]dto.QuestionAnswering, error) {
	optsOverride := s.opts.With(opts...)

	if req.Parameters != nil && req.Parameters.TopK != nil && *req.Parameters.TopK > 1 {
		return doJSONInference[dto.QuestionAnsweringRequest, []dto.QuestionAnswering](
			optsOverride,
			providers.TaskQuestionAnswering,
			req,
		)
	}

	single, err := doJSONInference[dto.QuestionAnsweringRequest, dto.QuestionAnswering](
		optsOverride,
		providers.TaskQuestionAnswering,
		req,
	)
	if err != nil {
		return nil, err
	}

	return []dto.QuestionAnswering{single}, nil
}
