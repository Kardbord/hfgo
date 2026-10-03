package hftypes

// QuestionAnsweringInput represents the input data for a question answering request.
// Both Question and Context are required.
type QuestionAnsweringInput struct {
	// The question to be answered.
	// Required.
	Question string `json:"question"`

	// The context to be used for answering the question.
	// Required.
	Context string `json:"context"`
}

// QuestionAnsweringRequest represents a question answering
// inference request to the API.
type QuestionAnsweringRequest struct {
	// The question and context pair to answer.
	// Required.
	Input QuestionAnsweringInput `json:"inputs"`

	// Additional inference parameters for question answering.
	Parameters *QuestionAnsweringParameters `json:"parameters,omitempty"`
}

// QuestionAnsweringParameters specify additional inference
// parameters for question answering.
type QuestionAnsweringParameters struct {
	// The number of answers to return (will be chosen by order of likelihood).
	// Note that less than top_k answers may be returned if there are not enough
	// options available within the context.
	TopK *int `json:"top_k,omitempty"`

	// If the context is too long to fit with the question for the model, it will
	// be split in several chunks with some overlap. This argument controls the
	// size of that overlap.
	DocStride *int `json:"doc_stride,omitempty"`

	// The maximum length of predicted answers (e.g., only answers with a shorter
	// length are considered).
	MaxAnswerLen *int `json:"max_answer_len,omitempty"`

	// The maximum length of the total sentence (context + question) in tokens of
	// each chunk passed to the model. The context will be split in several chunks
	// (using doc_stride as overlap) if needed.
	MaxSeqLen *int `json:"max_seq_len,omitempty"`

	// The maximum length of the question after tokenization. It will be truncated
	// if needed.
	MaxQuestionLen *int `json:"max_question_len,omitempty"`

	// Whether to accept impossible as an answer.
	HandleImpossibleAnswer *bool `json:"handle_impossible_answer,omitempty"`

	// Attempts to align the answer to real words. Improves quality on space
	// separated languages. Might hurt on non-space-separated languages (like
	// Japanese or Chinese).
	AlignToWords *bool `json:"align_to_words,omitempty"`
}

// QuestionAnswering represents a question answering output.
type QuestionAnswering struct {
	// The answer to the question.
	Answer string `json:"answer"`

	// The probability associated to the answer.
	Score float64 `json:"score"`

	// The character position in the input where the answer begins.
	Start int `json:"start"`

	// The character position in the input where the answer ends.
	End int `json:"end"`
}

// Clone returns a deep defensive copy of the request.
func (r *QuestionAnsweringRequest) Clone() QuestionAnsweringRequest {
	if r == nil {
		return QuestionAnsweringRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*QuestionAnsweringParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *QuestionAnsweringParameters) Clone() QuestionAnsweringParameters {
	if p == nil {
		return QuestionAnsweringParameters{}
	}
	out := *p
	out.TopK = clonePtr(p.TopK)
	out.DocStride = clonePtr(p.DocStride)
	out.MaxAnswerLen = clonePtr(p.MaxAnswerLen)
	out.MaxSeqLen = clonePtr(p.MaxSeqLen)
	out.MaxQuestionLen = clonePtr(p.MaxQuestionLen)
	out.HandleImpossibleAnswer = clonePtr(p.HandleImpossibleAnswer)
	out.AlignToWords = clonePtr(p.AlignToWords)

	return out
}
