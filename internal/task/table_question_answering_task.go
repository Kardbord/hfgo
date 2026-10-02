package task

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/providers"
)

// AnswerTableQuestion sends a table question answering request and returns the answer.
func AnswerTableQuestion(
	opts hfopts.Options,
	req hftypes.TableQuestionAnsweringRequest,
) (hftypes.TableQuestionAnswer, error) {
	return doJSONInference[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer](
		opts,
		providers.TaskTableQuestionAnswering,
		req,
	)
}
