package hftypes

import (
	"encoding/json"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ChatFunctionCall represents a tool function call with arguments.
type ChatFunctionCall struct {
	// Required.
	Name string `json:"name"`
	// Required.
	Arguments   string  `json:"arguments"`
	Description *string `json:"description,omitempty"`
}

// MarshalJSON enforces required fields on ChatFunctionCall.
func (f ChatFunctionCall) MarshalJSON() ([]byte, error) {
	if err := f.validate(); err != nil {
		return nil, err
	}
	type alias ChatFunctionCall

	return json.Marshal(alias(f))
}

// Clone returns a deep defensive copy of the function call.
func (f *ChatFunctionCall) Clone() ChatFunctionCall {
	if f == nil {
		return ChatFunctionCall{}
	}
	out := *f
	out.Description = clonePtr(f.Description)

	return out
}

func (f ChatFunctionCall) validate() error {
	if f.Name == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "chat function call: name must be set",
			Err:     nil,
		}
	}
	if f.Arguments == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "chat function call: arguments must be set",
			Err:     nil,
		}
	}

	return nil
}
