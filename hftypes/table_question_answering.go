package hftypes

import "maps"

// TableQuestionAnsweringInput represents the input data for a table
// question answering request. Both Question and Table are required.
type TableQuestionAnsweringInput struct {
	// The question to be answered about the table.
	// Required.
	Question string `json:"question"`

	// The table to serve as context for the questions.
	// Each key is a column name, and the value is a list of cell values for that column.
	// Required.
	Table map[string][]string `json:"table"`
}

// TableQuestionAnsweringRequest represents a table question answering
// inference request to the API.
type TableQuestionAnsweringRequest struct {
	// The question and table pair to answer.
	// Required.
	Input TableQuestionAnsweringInput `json:"inputs"`

	// Additional inference parameters for table question answering.
	Parameters *TableQuestionAnsweringParameters `json:"parameters,omitempty"`

	// ProviderParams is an advanced escape hatch for provider-specific
	// request parameters not covered by the typed fields. Supporting
	// providers merge these entries into the request body; entries override
	// typed fields. See the provider's documentation for supported
	// parameters. The map is treated as read-only while a request is in
	// flight.
	ProviderParams map[string]any `json:"-"`
}

// TableQuestionAnsweringParameters specify additional inference
// parameters for table question answering tasks.
type TableQuestionAnsweringParameters struct {
	// Activates and controls padding.
	Padding *string `json:"padding,omitempty"`

	// Whether to do inference sequentially or as a batch. Batching is faster,
	// but models like SQA require the inference to be done sequentially to
	// extract relations within sequences, given their conversational nature.
	Sequential *bool `json:"sequential,omitempty"`

	// Activates and controls truncation.
	Truncation *bool `json:"truncation,omitempty"`
}

const (
	// TableQuestionAnsweringPaddingDoNotPad does not pad the input.
	TableQuestionAnsweringPaddingDoNotPad = "do_not_pad"
	// TableQuestionAnsweringPaddingLongest pads to the longest sequence in the batch.
	TableQuestionAnsweringPaddingLongest = "longest"
	// TableQuestionAnsweringPaddingMaxLength pads to the maximum length.
	TableQuestionAnsweringPaddingMaxLength = "max_length"
)

// TableQuestionAnswer represents a table question answering output.
type TableQuestionAnswer struct {
	// The answer to the question given the table. If there is an aggregator,
	// the answer will be preceded by "AGGREGATOR >".
	Answer string `json:"answer"`

	// Coordinates of the cells of the answers.
	Coordinates [][]int `json:"coordinates"`

	// List of strings made up of the answer cell values.
	Cells []string `json:"cells"`

	// If the model has an aggregator, this returns the aggregator.
	Aggregator *string `json:"aggregator,omitempty"`

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
func (r *TableQuestionAnsweringRequest) Clone() TableQuestionAnsweringRequest {
	if r == nil {
		return TableQuestionAnsweringRequest{}
	}
	out := *r
	out.Input.Table = cloneStringSliceMap(r.Input.Table)
	out.Parameters = cloneStructPtr(r.Parameters, (*TableQuestionAnsweringParameters).Clone)
	out.ProviderParams = maps.Clone(r.ProviderParams)

	return out
}

// Clone returns a deep defensive copy of the parameters.
func (p *TableQuestionAnsweringParameters) Clone() TableQuestionAnsweringParameters {
	if p == nil {
		return TableQuestionAnsweringParameters{}
	}
	out := *p
	out.Padding = clonePtr(p.Padding)
	out.Sequential = clonePtr(p.Sequential)
	out.Truncation = clonePtr(p.Truncation)

	return out
}

// cloneStringSliceMap returns a deep copy of a map[string][]string.
func cloneStringSliceMap(src map[string][]string) map[string][]string {
	if src == nil {
		return nil
	}
	dst := maps.Clone(src)
	for k, v := range dst {
		dst[k] = append([]string(nil), v...)
	}

	return dst
}
