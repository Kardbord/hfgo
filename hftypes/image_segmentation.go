package hftypes

import "maps"

// ImageSegmentationRequest represents an image segmentation
// inference request to the API for a single input.
type ImageSegmentationRequest struct {
	// The input image data as a base64-encoded string.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for image segmentation.
	// Optional.
	Parameters *ImageSegmentationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// ImageSegmentationParameters specify additional inference parameters
// for image segmentation tasks.
type ImageSegmentationParameters struct {
	// Threshold to use when turning the predicted masks into binary values.
	MaskThreshold *float64 `json:"mask_threshold,omitempty"`

	// Mask overlap threshold to eliminate small, disconnected segments.
	OverlapMaskAreaThreshold *float64 `json:"overlap_mask_area_threshold,omitempty"`

	// Segmentation task to be performed, depending on model capabilities.
	Subtask ImageSegmentationSubtask `json:"subtask,omitempty"`

	// Probability threshold to filter out predicted masks.
	Threshold *float64 `json:"threshold,omitempty"`
}

// ImageSegmentationSubtask enumerates the segmentation subtasks a model may
// support.
type ImageSegmentationSubtask string

const (
	// ImageSegmentationSubtaskInstance segments individual objects.
	ImageSegmentationSubtaskInstance ImageSegmentationSubtask = "instance"
	// ImageSegmentationSubtaskPanoptic segments both countable objects and
	// uncountable regions.
	ImageSegmentationSubtaskPanoptic ImageSegmentationSubtask = "panoptic"
	// ImageSegmentationSubtaskSemantic segments pixels by class without
	// distinguishing instances.
	ImageSegmentationSubtaskSemantic ImageSegmentationSubtask = "semantic"
)

// ImageSegmentation represents a predicted mask / segment.
type ImageSegmentation struct {
	// The label of the predicted segment.
	Label string `json:"label"`

	// The corresponding mask as a black-and-white image (base64-encoded).
	Mask string `json:"mask"`

	// The score or confidence degree the model has.
	Score *float64 `json:"score,omitempty"`

	// ProviderMetadata is an advanced escape hatch for provider-specific
	// response metadata not covered by the typed fields. Providers that
	// support it populate the map with additional upstream response data;
	// it stays nil otherwise. Generic JSON decoding does not populate it:
	// only provider-specific codecs do.
	ProviderMetadata map[string]any `json:"-"`
}

// Clone returns a deep defensive copy of the request. The ProviderParams map
// is copied as a new map, but its values are shared because their types are
// not known statically.
func (r *ImageSegmentationRequest) Clone() ImageSegmentationRequest {
	if r == nil {
		return ImageSegmentationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ImageSegmentationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *ImageSegmentationParameters) Clone() ImageSegmentationParameters {
	if p == nil {
		return ImageSegmentationParameters{}
	}
	out := *p
	out.MaskThreshold = clonePtr(p.MaskThreshold)
	out.OverlapMaskAreaThreshold = clonePtr(p.OverlapMaskAreaThreshold)
	out.Threshold = clonePtr(p.Threshold)

	return out
}
