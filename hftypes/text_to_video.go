package hftypes

import "maps"

// TextToVideoResponse is a generated video together with the media type the
// API advertised for it.
//
// Unlike the request and parameter DTOs, this value is assembled by the
// text-to-video codec from the raw response body and Content-Type header
// rather than unmarshaled from JSON, so its fields are excluded from JSON
// encoding.
type TextToVideoResponse struct {
	// Video is the raw video bytes (for example MP4 data).
	Video []byte `json:"-"`

	// MediaType is the parsed, normalized MIME type of Video (for example
	// "video/mp4").
	MediaType string `json:"-"`

	// ProviderMetadata is an advanced escape hatch for provider-specific
	// response metadata not covered by the typed fields. Providers that
	// support it populate the map with additional upstream response data
	// (for example task IDs or timing information); it stays nil otherwise.
	ProviderMetadata map[string]any `json:"-"`
}

// TextToVideoRequest represents a text-to-video
// inference request to the API for a single input.
type TextToVideoRequest struct {
	// The input text data (sometimes called "prompt").
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for text-to-video.
	Parameters *TextToVideoParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// TextToVideoParameters specify additional inference
// parameters for text-to-video tasks.
type TextToVideoParameters struct {
	// The num_frames parameter determines how many video frames are generated.
	NumFrames *int `json:"num_frames,omitempty"`

	// A higher guidance scale value encourages the model to generate videos
	// closely linked to the text prompt, but values too high may cause
	// saturation and other artifacts.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`

	// One or several prompt to guide what NOT to include in video generation.
	NegativePrompt *[]string `json:"negative_prompt,omitempty"`

	// The number of denoising steps. More denoising steps usually lead to a
	// higher quality video at the expense of slower inference.
	NumInferenceSteps *int `json:"num_inference_steps,omitempty"`

	// Seed for the random number generator.
	Seed *int64 `json:"seed,omitempty"`
}

// Clone returns a deep defensive copy of the request. The ProviderParams map
// is copied as a new map, but its values are shared because their types are
// not known statically.
func (r *TextToVideoRequest) Clone() TextToVideoRequest {
	if r == nil {
		return TextToVideoRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TextToVideoParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *TextToVideoParameters) Clone() TextToVideoParameters {
	if p == nil {
		return TextToVideoParameters{}
	}
	out := *p
	out.NumFrames = clonePtr(p.NumFrames)
	out.GuidanceScale = clonePtr(p.GuidanceScale)
	if p.NegativePrompt != nil {
		neg := cloneSlice(*p.NegativePrompt, func(s *string) string { return *s })
		out.NegativePrompt = &neg
	}
	out.NumInferenceSteps = clonePtr(p.NumInferenceSteps)
	out.Seed = clonePtr(p.Seed)

	return out
}
