package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	SummarizationRequest      = dto.SummarizationRequest
	SummarizationBatchRequest = dto.SummarizationBatchRequest
	SummarizationParameters   = dto.SummarizationParameters
	Summarization             = dto.Summarization
)

//nolint:revive // constants forward docs from internal/dto
const (
	SummarizationTruncationDoNotTruncate = dto.SummarizationTruncationDoNotTruncate
	SummarizationTruncationLongestFirst  = dto.SummarizationTruncationLongestFirst
	SummarizationTruncationOnlyFirst     = dto.SummarizationTruncationOnlyFirst
	SummarizationTruncationOnlySecond    = dto.SummarizationTruncationOnlySecond
)
