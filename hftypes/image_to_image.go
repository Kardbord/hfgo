package hftypes

// ImageToImageResponse is a generated image together with the media type the
// API advertised for it.
//
// Unlike the request and parameter DTOs, this value is assembled by the
// image-to-image codec from the raw response body and Content-Type header
// rather than unmarshaled from JSON, so its fields are excluded from JSON
// encoding.
type ImageToImageResponse struct {
	// Image is the raw image bytes (for example PNG or JPEG data).
	Image []byte `json:"-"`

	// MediaType is the parsed, normalized MIME type of Image (for example
	// "image/png").
	MediaType string `json:"-"`
}

// ImageToImageRequest represents an image-to-image inference request to the
// API for a single input.
type ImageToImageRequest struct {
	// The input image data as a base64-encoded string. If no parameters are
	// provided, you can also provide the image data as a raw bytes payload.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for image-to-image.
	Parameters *ImageToImageParameters `json:"parameters,omitempty"`
}

// ImageToImageParameters specify additional inference parameters for
// image-to-image tasks.
type ImageToImageParameters struct {
	// The text prompt to guide the image generation.
	Prompt *string `json:"prompt,omitempty"`

	// A higher guidance scale value encourages the model to generate images
	// closely linked to the text prompt at the expense of lower image quality.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`

	// One prompt to guide what NOT to include in image generation.
	NegativePrompt *string `json:"negative_prompt,omitempty"`

	// The number of denoising steps. More denoising steps usually lead to a
	// higher quality image at the expense of slower inference.
	NumInferenceSteps *int `json:"num_inference_steps,omitempty"`

	// The size in pixels of the output image. This parameter is only supported
	// by some providers and for specific models. It will be ignored when
	// unsupported.
	TargetSize *ImageToImageTargetSize `json:"target_size,omitempty"`
}

// ImageToImageTargetSize specifies the size in pixels of the output image.
type ImageToImageTargetSize struct {
	// The width in pixels.
	// Required.
	Width *int `json:"width,omitempty"`

	// The height in pixels.
	// Required.
	Height *int `json:"height,omitempty"`
}

// Clone returns a deep defensive copy of the request.
func (r *ImageToImageRequest) Clone() ImageToImageRequest {
	if r == nil {
		return ImageToImageRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*ImageToImageParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *ImageToImageParameters) Clone() ImageToImageParameters {
	if p == nil {
		return ImageToImageParameters{}
	}
	out := *p
	out.Prompt = clonePtr(p.Prompt)
	out.GuidanceScale = clonePtr(p.GuidanceScale)
	out.NegativePrompt = clonePtr(p.NegativePrompt)
	out.NumInferenceSteps = clonePtr(p.NumInferenceSteps)
	out.TargetSize = cloneStructPtr(p.TargetSize, (*ImageToImageTargetSize).Clone)

	return out
}

// Clone returns a deep defensive copy of the target size.
func (s *ImageToImageTargetSize) Clone() ImageToImageTargetSize {
	if s == nil {
		return ImageToImageTargetSize{}
	}
	out := *s
	out.Width = clonePtr(s.Width)
	out.Height = clonePtr(s.Height)

	return out
}
