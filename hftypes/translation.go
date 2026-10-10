package hftypes

import "maps"

// TranslationRequest represents a translation
// inference request to the API for a single input.
type TranslationRequest struct {
	// The text to translate.
	// Required.
	Input string `json:"inputs"`

	// Additional inference parameters for translation.
	Parameters *TranslationParameters `json:"parameters,omitempty"`
}

// TranslationParameters specify additional inference
// parameters for translation tasks.
type TranslationParameters struct {
	// Whether to clean up the potential extra spaces in the text output.
	CleanUpTokenizationSpaces *bool `json:"clean_up_tokenization_spaces,omitempty"`

	// The source language of the text. Required for models that can
	// translate from multiple languages.
	SrcLang *string `json:"src_lang,omitempty"`

	// The target language to translate to. Required for models that can
	// translate to multiple languages.
	TgtLang *string `json:"tgt_lang,omitempty"`

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
	// TranslationTruncationDoNotTruncate keeps the input as-is without any truncation.
	TranslationTruncationDoNotTruncate = "do_not_truncate"
	// TranslationTruncationLongestFirst truncates the longest side first when the input exceeds the model's maximum length.
	TranslationTruncationLongestFirst = "longest_first"
	// TranslationTruncationOnlyFirst trims the input on the first (left) side when it exceeds the model's maximum length.
	TranslationTruncationOnlyFirst = "only_first"
	// TranslationTruncationOnlySecond trims the input on the second (right) side when it exceeds the model's maximum length.
	TranslationTruncationOnlySecond = "only_second"
)

// Translation represents a translation output.
type Translation struct {
	// The translated text.
	TranslationText string `json:"translation_text"`
}

// Clone returns a deep defensive copy of the request.
func (r *TranslationRequest) Clone() TranslationRequest {
	if r == nil {
		return TranslationRequest{}
	}
	out := *r
	out.Parameters = cloneStructPtr(r.Parameters, (*TranslationParameters).Clone)

	return out
}

// Clone returns a deep defensive copy of the parameters. The GenerateParameters
// map is copied as a new map, but its values are shared because their types are
// not known statically.
func (p *TranslationParameters) Clone() TranslationParameters {
	if p == nil {
		return TranslationParameters{}
	}
	out := *p
	out.CleanUpTokenizationSpaces = clonePtr(p.CleanUpTokenizationSpaces)
	out.SrcLang = clonePtr(p.SrcLang)
	out.TgtLang = clonePtr(p.TgtLang)
	out.Truncation = clonePtr(p.Truncation)
	out.GenerateParameters = maps.Clone(p.GenerateParameters)

	return out
}
