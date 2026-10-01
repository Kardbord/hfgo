package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	TableQuestionAnsweringInput      = dto.TableQuestionAnsweringInput
	TableQuestionAnsweringRequest    = dto.TableQuestionAnsweringRequest
	TableQuestionAnsweringParameters = dto.TableQuestionAnsweringParameters
	TableQuestionAnswer              = dto.TableQuestionAnswer
)

//nolint:revive // constants forward docs from internal/dto
const (
	TableQuestionAnsweringPaddingDoNotPad  = dto.TableQuestionAnsweringPaddingDoNotPad
	TableQuestionAnsweringPaddingLongest   = dto.TableQuestionAnsweringPaddingLongest
	TableQuestionAnsweringPaddingMaxLength = dto.TableQuestionAnsweringPaddingMaxLength
)
