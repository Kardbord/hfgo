package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// AnswerQuestion sends a question answering request and returns the answers.
// When TopK > 1 it performs a multi-answer inference; otherwise it wraps
// the single-answer response into a one-element list.
func AnswerQuestion(
	opts hfopts.Options,
	req hftypes.QuestionAnsweringRequest,
) ([]hftypes.QuestionAnswering, error) {
	if req.Parameters != nil && req.Parameters.TopK != nil && *req.Parameters.TopK > 1 {
		return doJSONInference[hftypes.QuestionAnsweringRequest, []hftypes.QuestionAnswering](
			opts,
			hfproviders.TaskQuestionAnswering,
			req,
		)
	}

	single, err := doJSONInference[hftypes.QuestionAnsweringRequest, hftypes.QuestionAnswering](
		opts,
		hfproviders.TaskQuestionAnswering,
		req,
	)
	if err != nil {
		return nil, err
	}

	return []hftypes.QuestionAnswering{single}, nil
}
