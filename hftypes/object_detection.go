package hftypes

import "maps"

// ObjectDetectionRequest represents an object detection
// inference request to the API for a single input.
type ObjectDetectionRequest struct {
	// The input image data as a base64-encoded string.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for object detection.
	// Optional.
	Parameters *ObjectDetectionParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// ObjectDetectionParameters specify additional inference parameters
// for object detection tasks.
type ObjectDetectionParameters struct {
	// The probability necessary to make a prediction.
	Threshold *float64 `json:"threshold,omitempty"`
}

// ObjectDetection represents an object detection task output.
type ObjectDetection struct {
	// The predicted label for the bounding box.
	Label string `json:"label"`

	// The associated score / probability.
	Score float64 `json:"score"`

	// The bounding box for the detected object.
	Box ObjectDetectionBoundingBox `json:"box"`

	// ProviderMetadata is an advanced escape hatch for provider-specific
	// response metadata not covered by the typed fields. Providers that
	// support it populate the map with additional upstream response data;
	// it stays nil otherwise. Generic JSON decoding does not populate it:
	// only provider-specific codecs do.
	ProviderMetadata map[string]any `json:"-"`
}

// ObjectDetectionBoundingBox defines the bounds of
// a detected object within an image.
type ObjectDetectionBoundingBox struct {
	// The x-coordinate of the top-left corner of the bounding box.
	XMin int `json:"xmin"`

	// The x-coordinate of the bottom-right corner of the bounding box.
	XMax int `json:"xmax"`

	// The y-coordinate of the top-left corner of the bounding box.
	YMin int `json:"ymin"`

	// The y-coordinate of the bottom-right corner of the bounding box.
	YMax int `json:"ymax"`
}

// Clone returns a deep defensive copy of the request. The ProviderParams map
// is copied as a new map, but its values are shared because their types are
// not known statically.
func (r *ObjectDetectionRequest) Clone() ObjectDetectionRequest {
	if r == nil {
		return ObjectDetectionRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ObjectDetectionParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *ObjectDetectionParameters) Clone() ObjectDetectionParameters {
	if p == nil {
		return ObjectDetectionParameters{}
	}
	out := *p
	out.Threshold = clonePtr(p.Threshold)

	return out
}
