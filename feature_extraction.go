package hfgo

import "github.com/Kardbord/hfgo/v4/internal/dto"

//nolint:revive // type aliases forward docs from internal/dto
type (
	FeatureExtraction             = dto.FeatureExtraction
	FeatureExtractionRequest      = dto.FeatureExtractionRequest
	FeatureExtractionBatchRequest = dto.FeatureExtractionBatchRequest
	FeatureExtractionParameters   = dto.FeatureExtractionParameters
)

//nolint:revive // constants forward docs from internal/dto
const (
	FeatureExtractionTruncationLeft  = dto.FeatureExtractionTruncationLeft
	FeatureExtractionTruncationRight = dto.FeatureExtractionTruncationRight
)
