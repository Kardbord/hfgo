package hftypes

// ImageSegmentationRequest represents an image segmentation
// inference request to the API for a single input.
type ImageSegmentationRequest struct {
	// The input image data as a base64-encoded string.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for image segmentation.
	// Optional.
	Parameters *ImageSegmentationParameters `json:"parameters,omitempty"`
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
}

// Clone returns a deep defensive copy of the request.
func (r *ImageSegmentationRequest) Clone() ImageSegmentationRequest {
	if r == nil {
		return ImageSegmentationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ImageSegmentationParameters).Clone)

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
