package hftypes

import "maps"

// SummarizationRequest represents a summarization
// inference request to the API for a single input.
type SummarizationRequest struct {
	// The text to summarize.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for summarization.
	Parameters *SummarizationParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// SummarizationParameters specify additional inference
// parameters for summarization tasks.
type SummarizationParameters struct {
	// Whether to clean up the potential extra spaces in the text output.
	CleanUpTokenizationSpaces *bool `json:"clean_up_tokenization_spaces,omitempty"`

	// The truncation strategy to use.
	Truncation *string `json:"truncation,omitempty"`

	// GenerateParameters provides additional parametrization of the text
	// generation algorithm (e.g. "max_new_tokens", "temperature", "top_k").
	//
	// It is exposed because it is part of the upstream inference schema, but the
	// set of valid arguments is not documented by Hugging Face and may depend on
	// the model being used. Invalid or unsupported arguments can be rejected by
	// the API.
	GenerateParameters map[string]any `json:"generate_parameters,omitempty"`
}

const (
	// SummarizationTruncationDoNotTruncate keeps the input as-is without any truncation.
	SummarizationTruncationDoNotTruncate = "do_not_truncate"
	// SummarizationTruncationLongestFirst truncates the longest side first when the input exceeds the model's maximum length.
	SummarizationTruncationLongestFirst = "longest_first"
	// SummarizationTruncationOnlyFirst trims the input on the first (left) side when it exceeds the model's maximum length.
	SummarizationTruncationOnlyFirst = "only_first"
	// SummarizationTruncationOnlySecond trims the input on the second (right) side when it exceeds the model's maximum length.
	SummarizationTruncationOnlySecond = "only_second"
)

// Summarization represents a summarization output.
type Summarization struct {
	// The summarized text.
	SummaryText string `json:"summary_text"`

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
func (r *SummarizationRequest) Clone() SummarizationRequest {
	if r == nil {
		return SummarizationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*SummarizationParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters. The GenerateParameters
// map is copied as a new map, but its values are shared because their types are
// not known statically.
func (p *SummarizationParameters) Clone() SummarizationParameters {
	if p == nil {
		return SummarizationParameters{}
	}
	out := *p
	out.CleanUpTokenizationSpaces = clonePtr(p.CleanUpTokenizationSpaces)
	out.Truncation = clonePtr(p.Truncation)
	out.GenerateParameters = maps.Clone(p.GenerateParameters)

	return out
}
