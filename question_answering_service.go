package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// questionAnsweringService implements question answering calls using the configured request options.
type questionAnsweringService struct {
	opts request.Options
}

// newQuestionAnsweringService builds a question answering service with a snapshot of the provided options.
func newQuestionAnsweringService(opts request.Options) questionAnsweringService {
	return questionAnsweringService{opts: opts}
}

// answer sends a question answering request and returns the answers.
func (s questionAnsweringService) answer(
	req QuestionAnsweringRequest,
	opts ...Option,
) ([]QuestionAnswering, error) {
	optsOverride := s.opts.With(opts...)

	// NOTE: The HuggingFace API returns different response formats depending on
	// whether the top_k parameter is set:
	//
	//   - When top_k is unset (nil) or 1: Returns a bare JSON object
	//     {"answer":"...","score":...,"start":...,"end":...}
	//
	//   - When top_k is set (>1): Returns a JSON array
	//     [{"answer":"...","score":...},...]
	//
	// We handle this by unmarshaling into the appropriate type based on the
	// request parameters, similar to how text classification handles its
	// TopK response format quirk.
	if req.Parameters != nil && req.Parameters.TopK != nil && *req.Parameters.TopK > 1 {
		return doModelInference[QuestionAnsweringRequest, []QuestionAnswering](
			optsOverride,
			TaskQuestionAnswering,
			req,
		)
	}

	single, err := doModelInference[QuestionAnsweringRequest, QuestionAnswering](
		optsOverride,
		TaskQuestionAnswering,
		req,
	)
	if err != nil {
		return nil, err
	}

	return []QuestionAnswering{single}, nil
}
