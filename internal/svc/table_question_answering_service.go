package svc

import (
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// TableQuestionAnsweringService implements table question answering calls.
type TableQuestionAnsweringService struct {
	opts request.Options
}

// NewTableQuestionAnsweringService builds a table QA service.
func NewTableQuestionAnsweringService(opts request.Options) TableQuestionAnsweringService {
	return TableQuestionAnsweringService{opts: opts}
}

// Answer sends a table question answering request and returns the answer.
func (s TableQuestionAnsweringService) Answer(
	req dto.TableQuestionAnsweringRequest,
	opts ...request.Option,
) (dto.TableQuestionAnswer, error) {
	return doJSONInference[dto.TableQuestionAnsweringRequest, dto.TableQuestionAnswer](
		s.opts.With(opts...),
		providers.TaskTableQuestionAnswering,
		req,
	)
}
