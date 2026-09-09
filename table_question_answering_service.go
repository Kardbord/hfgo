package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// tableQuestionAnsweringService implements table question answering calls using the configured request options.
type tableQuestionAnsweringService struct {
	opts request.Options
}

// newTableQuestionAnsweringService builds a table question answering service with a snapshot of the provided options.
func newTableQuestionAnsweringService(opts request.Options) tableQuestionAnsweringService {
	return tableQuestionAnsweringService{opts: opts}
}

// answer sends a table question answering request and returns the answer.
//
// NOTE: The HuggingFace API returns a bare JSON object for table question
// answering, not an array — despite the upstream schema declaring an array
// response. We defer to the actual API behavior and return a single
// TableQuestionAnswer.
func (s tableQuestionAnsweringService) answer(
	req TableQuestionAnsweringRequest,
	opts ...Option,
) (TableQuestionAnswer, error) {
	return doModelInference[TableQuestionAnsweringRequest, TableQuestionAnswer](
		s.opts.With(opts...),
		TaskTableQuestionAnswering,
		req,
	)
}
