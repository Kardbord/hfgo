package task

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// AnswerTableQuestion sends a table question answering request and returns the answer.
func AnswerTableQuestion(
	opts request.Options,
	req dto.TableQuestionAnsweringRequest,
) (dto.TableQuestionAnswer, error) {
	return doJSONInference[dto.TableQuestionAnsweringRequest, dto.TableQuestionAnswer](
		opts,
		providers.TaskTableQuestionAnswering,
		req,
	)
}
