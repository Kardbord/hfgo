package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	TranslationRequest      = dto.TranslationRequest
	TranslationBatchRequest = dto.TranslationBatchRequest
	TranslationParameters   = dto.TranslationParameters
	Translation             = dto.Translation
)

//nolint:revive // constants forward docs from internal/dto
const (
	TranslationTruncationDoNotTruncate = dto.TranslationTruncationDoNotTruncate
	TranslationTruncationLongestFirst  = dto.TranslationTruncationLongestFirst
	TranslationTruncationOnlyFirst     = dto.TranslationTruncationOnlyFirst
	TranslationTruncationOnlySecond    = dto.TranslationTruncationOnlySecond
)
