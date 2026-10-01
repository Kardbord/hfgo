package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	TextClassificationRequest      = dto.TextClassificationRequest
	TextClassificationBatchRequest = dto.TextClassificationBatchRequest
	TextClassificationParameters   = dto.TextClassificationParameters
	TextClassification             = dto.TextClassification
)

//nolint:revive // constants forward docs from internal/dto
const (
	TextClassificationFuncSigmoid = dto.TextClassificationFuncSigmoid
	TextClassificationFuncSoftmax = dto.TextClassificationFuncSoftmax
	TextClassificationFuncNone    = dto.TextClassificationFuncNone
)
