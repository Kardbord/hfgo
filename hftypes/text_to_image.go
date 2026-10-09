package hftypes

// TextToImageResponse is a generated image together with the media type the
// API advertised for it.
//
// Unlike the request and parameter DTOs, this value is assembled by the
// text-to-image codec from the raw response body and Content-Type header
// rather than unmarshaled from JSON, so its fields are excluded from JSON
// encoding.
type TextToImageResponse struct {
	// Image is the raw image bytes (for example PNG or JPEG data).
	Image []byte `json:"-"`

	// MediaType is the parsed, normalized MIME type of Image (for example
	// "image/png").
	MediaType string `json:"-"`
}

// TextToImageRequest represents a text-to-image
// inference request to the API for a single input.
type TextToImageRequest struct {
	// The input text data (sometimes called "prompt").
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for text-to-image.
	Parameters *TextToImageParameters `json:"parameters,omitempty"`
}

// TextToImageParameters specify additional inference
// parameters for text-to-image tasks.
type TextToImageParameters struct {
	// A higher guidance scale value encourages the model to generate images
	// closely linked to the text prompt, but values too high may cause
	// saturation and other artifacts.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`

	// One prompt to guide what NOT to include in image generation.
	NegativePrompt *string `json:"negative_prompt,omitempty"`

	// The number of denoising steps. More denoising steps usually lead to a
	// higher quality image at the expense of slower inference.
	NumInferenceSteps *int `json:"num_inference_steps,omitempty"`

	// The width in pixels of the output image.
	Width *int `json:"width,omitempty"`

	// The height in pixels of the output image.
	Height *int `json:"height,omitempty"`

	// Override the scheduler with a compatible one.
	Scheduler *string `json:"scheduler,omitempty"`

	// Seed for the random number generator.
	Seed *int64 `json:"seed,omitempty"`
}

// Clone returns a deep defensive copy of the request.
func (r *TextToImageRequest) Clone() TextToImageRequest {
	if r == nil {
		return TextToImageRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TextToImageParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *TextToImageParameters) Clone() TextToImageParameters {
	if p == nil {
		return TextToImageParameters{}
	}
	out := *p
	out.GuidanceScale = clonePtr(p.GuidanceScale)
	out.NegativePrompt = clonePtr(p.NegativePrompt)
	out.NumInferenceSteps = clonePtr(p.NumInferenceSteps)
	out.Width = clonePtr(p.Width)
	out.Height = clonePtr(p.Height)
	out.Scheduler = clonePtr(p.Scheduler)
	out.Seed = clonePtr(p.Seed)

	return out
}
