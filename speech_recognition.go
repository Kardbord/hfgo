package hfgo

import (
	"encoding/json"
	"fmt"
	"slices"
)

// SpeechRecognitionRequest represents an automatic speech recognition
// inference request to the API for a single input.
type SpeechRecognitionRequest struct {
	// The input audio data as a base64-encoded string.
	Input string `json:"input"`

	// Additional inference parameters for Automatic Speech Recognition.
	Parameters *SpeechRecognitionRequestParameters `json:"parameters,omitempty"`
}

// SpeechRecognitionBatchRequest represents a batched automatic speech
// recognition inference request to the API for multiple inputs.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
type SpeechRecognitionBatchRequest struct {
	// The input audio data as base64-encoded strings.
	Inputs []string `json:"inputs"`

	// Additional inference parameters for Automatic Speech Recognition.
	Parameters *SpeechRecognitionRequestParameters `json:"parameters,omitempty"`
}

// SpeechRecognitionRequestParameters specify additional inference
// parameters for automatic speech recognition tasks.
type SpeechRecognitionRequestParameters struct {
	// Whether to output corresponding timestamps with the generated text.
	ReturnTimestamps *bool `json:"return_timestamps,omitempty"`

	// Parameterization of the text generation process.
	GenerationParameters *SpeechRecognitionGenerationParameters `json:"generation_parameters,omitempty"`
}

// SpeechRecognitionGenerationParameters represents parameterizations of
// the speech recognition text generation process.
type SpeechRecognitionGenerationParameters struct {
	Temperature   *float64       `json:"temperature,omitempty"`
	TopK          *int           `json:"top_k,omitempty"`
	TopP          *float64       `json:"top_p,omitempty"`
	TypicalP      *float64       `json:"typical_p,omitempty"`
	EpsilonCutoff *float64       `json:"epsilon_cutoff,omitempty"`
	EtaCutoff     *float64       `json:"eta_cutoff,omitempty"`
	MaxLength     *int           `json:"max_length,omitempty"`
	MaxNewTokens  *int           `json:"max_new_tokens,omitempty"`
	MinLength     *int           `json:"min_length,omitempty"`
	MinNewTokens  *int           `json:"min_new_tokens,omitempty"`
	DoSample      *bool          `json:"do_sample,omitempty"`
	EarlyStopping *EarlyStopping `json:"early_stopping,omitempty"`
	NumBeams      *int           `json:"num_beams,omitempty"`
	NumBeamGroups *int           `json:"num_beam_groups,omitempty"`
	PenaltyAlpha  *float64       `json:"penalty_alpha,omitempty"`
	UseCache      *bool          `json:"use_cache,omitempty"`
}

// EarlyStopping represents the early_stopping parameter for speech
// recognition generation. It accepts true, false, or the string "never".
type EarlyStopping string

// EarlyStoppingTrue enables early stopping.
const EarlyStoppingTrue EarlyStopping = "true"

// EarlyStoppingFalse disables early stopping.
const EarlyStoppingFalse EarlyStopping = "false"

// EarlyStoppingNever disables early stopping with the "never" string value.
const EarlyStoppingNever EarlyStopping = "never"

// MarshalJSON implements json.Marshaler for EarlyStopping.
func (e EarlyStopping) MarshalJSON() ([]byte, error) {
	switch e {
	case EarlyStoppingTrue:
		return json.Marshal(true)
	case EarlyStoppingFalse:
		return json.Marshal(false)
	case EarlyStoppingNever:
		return json.Marshal("never")
	default:
		return json.Marshal(string(e))
	}
}

// UnmarshalJSON implements json.Unmarshaler for EarlyStopping.
func (e *EarlyStopping) UnmarshalJSON(data []byte) error {
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		if b {
			*e = EarlyStoppingTrue
		} else {
			*e = EarlyStoppingFalse
		}

		return nil
	}

	var str string
	err := json.Unmarshal(data, &str)
	if err == nil {
		switch str {
		case "true":
			*e = EarlyStoppingTrue

			return nil
		case "false":
			*e = EarlyStoppingFalse

			return nil
		case "never":
			*e = EarlyStoppingNever

			return nil
		default:
			return fmt.Errorf("invalid EarlyStopping value %q", str)
		}
	}

	return err
}

// SpeechRecognition represents a single speech recognition output.
type SpeechRecognition struct {
	// The recognized text.
	Text string `json:"text"`

	// When return_timestamps is enabled, chunks contains a list of
	// audio chunks identified by the model.
	Chunks []SpeechRecognitionChunk `json:"chunks,omitempty"`
}

// SpeechRecognitionChunk is a single chunk of recognized speech with
// an associated timestamp range.
type SpeechRecognitionChunk struct {
	// A chunk of text identified by the model.
	Text string `json:"text"`

	// The start and end timestamps corresponding with the text.
	Timestamp []float64 `json:"timestamp"`
}

// Clone returns a deep defensive copy of the request.
func (r *SpeechRecognitionRequest) Clone() SpeechRecognitionRequest {
	if r == nil {
		return SpeechRecognitionRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*SpeechRecognitionRequestParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the request.
func (r *SpeechRecognitionBatchRequest) Clone() SpeechRecognitionBatchRequest {
	if r == nil {
		return SpeechRecognitionBatchRequest{}
	}
	out := *r
	out.Inputs = slices.Clone(r.Inputs)
	out.Parameters = cloneStructPtr(r.Parameters, (*SpeechRecognitionRequestParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *SpeechRecognitionRequestParameters) Clone() SpeechRecognitionRequestParameters {
	if p == nil {
		return SpeechRecognitionRequestParameters{}
	}
	out := *p
	out.ReturnTimestamps = clonePtr(p.ReturnTimestamps)
	out.GenerationParameters = cloneStructPtr(p.GenerationParameters,
		(*SpeechRecognitionGenerationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *SpeechRecognitionGenerationParameters) Clone() SpeechRecognitionGenerationParameters {
	if p == nil {
		return SpeechRecognitionGenerationParameters{}
	}
	out := *p
	out.Temperature = clonePtr(p.Temperature)
	out.TopK = clonePtr(p.TopK)
	out.TopP = clonePtr(p.TopP)
	out.TypicalP = clonePtr(p.TypicalP)
	out.EpsilonCutoff = clonePtr(p.EpsilonCutoff)
	out.EtaCutoff = clonePtr(p.EtaCutoff)
	out.MaxLength = clonePtr(p.MaxLength)
	out.MaxNewTokens = clonePtr(p.MaxNewTokens)
	out.MinLength = clonePtr(p.MinLength)
	out.MinNewTokens = clonePtr(p.MinNewTokens)
	out.DoSample = clonePtr(p.DoSample)
	out.EarlyStopping = clonePtr(p.EarlyStopping)
	out.NumBeams = clonePtr(p.NumBeams)
	out.NumBeamGroups = clonePtr(p.NumBeamGroups)
	out.PenaltyAlpha = clonePtr(p.PenaltyAlpha)
	out.UseCache = clonePtr(p.UseCache)

	return out
}
