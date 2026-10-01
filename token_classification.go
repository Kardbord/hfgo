package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	TokenClassificationRequest      = dto.TokenClassificationRequest
	TokenClassificationBatchRequest = dto.TokenClassificationBatchRequest
	TokenClassificationParameters   = dto.TokenClassificationParameters
	TokenClassification             = dto.TokenClassification
)

//nolint:revive // constants forward docs from internal/dto
const (
	TokenClassificationAggregationNone    = dto.TokenClassificationAggregationNone
	TokenClassificationAggregationSimple  = dto.TokenClassificationAggregationSimple
	TokenClassificationAggregationFirst   = dto.TokenClassificationAggregationFirst
	TokenClassificationAggregationAverage = dto.TokenClassificationAggregationAverage
	TokenClassificationAggregationMax     = dto.TokenClassificationAggregationMax
)
