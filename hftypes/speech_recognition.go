package hftypes

import "maps"

// SpeechRecognitionRequest represents an automatic speech recognition
// inference request.
type SpeechRecognitionRequest struct {
	// The input audio data as a base64-encoded string.
	// Required.
	//
	// When Parameters are omitted, the audio is sent as raw bytes. The
	// Content-Type is auto-detected from the decoded bytes via magic-byte
	// sniffing. To use a specific Content-Type instead, set it via
	// hfopts.WithHeader, e.g. hfopts.WithHeader("Content-Type", "audio/wav").
	// Caller-supplied headers always take precedence over auto-detected values.
	Input string `json:"inputs"`

	// Additional inference parameters for speech recognition.
	// Optional.
	Parameters *SpeechRecognitionParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// SpeechRecognitionParameters specify additional inference parameters
// for speech recognition tasks.
type SpeechRecognitionParameters struct {
	// When true, the response includes timestamps at the chunk level
	// in addition to the full transcript.
	ReturnTimestamps *bool `json:"return_timestamps,omitempty"`
}

// SpeechRecognitionResponse represents the output of a speech recognition
// inference request.
type SpeechRecognitionResponse struct {
	// The recognized text.
	Text string `json:"text"`

	// Chunk-level transcription with optional timestamps.
	// Present only when return_timestamps is true.
	Chunks []SpeechRecognitionChunk `json:"chunks,omitempty"`

	// ProviderMetadata is an advanced escape hatch for provider-specific
	// response metadata not covered by the typed fields. Providers that
	// support it populate the map with additional upstream response data;
	// it stays nil otherwise. Generic JSON decoding does not populate it:
	// only provider-specific codecs do.
	ProviderMetadata map[string]any `json:"-"`
}

// SpeechRecognitionChunk represents a segment of the transcription with
// optional timestamps.
type SpeechRecognitionChunk struct {
	// The recognized text for this chunk.
	Text string `json:"text"`

	// The start and end timestamps in seconds, as a two-element array
	// [start, end].
	Timestamp []float64 `json:"timestamp"`
}

// Clone returns a deep defensive copy of the request. The ProviderParams map
// is copied as a new map, but its values are shared because their types are
// not known statically.
func (r *SpeechRecognitionRequest) Clone() SpeechRecognitionRequest {
	if r == nil {
		return SpeechRecognitionRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*SpeechRecognitionParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *SpeechRecognitionParameters) Clone() SpeechRecognitionParameters {
	if p == nil {
		return SpeechRecognitionParameters{}
	}
	out := *p
	out.ReturnTimestamps = clonePtr(p.ReturnTimestamps)

	return out
}
