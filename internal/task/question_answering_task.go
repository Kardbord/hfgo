package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// AnswerQuestion sends a question answering request and returns the answers.
// When TopK > 1 it performs a multi-answer inference; otherwise it wraps
// the single-answer response into a one-element list.
func AnswerQuestion(
	opts request.Options,
	req dto.QuestionAnsweringRequest,
) ([]dto.QuestionAnswering, error) {
	if req.Parameters != nil && req.Parameters.TopK != nil && *req.Parameters.TopK > 1 {
		return doJSONInference[dto.QuestionAnsweringRequest, []dto.QuestionAnswering](
			opts,
			providers.TaskQuestionAnswering,
			req,
		)
	}

	single, err := doJSONInference[dto.QuestionAnsweringRequest, dto.QuestionAnswering](
		opts,
		providers.TaskQuestionAnswering,
		req,
	)
	if err != nil {
		return nil, err
	}

	return []dto.QuestionAnswering{single}, nil
}
