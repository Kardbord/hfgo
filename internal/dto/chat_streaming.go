package dto

import (
	"encoding/json"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
)

// ChatStreamResponse represents a streaming response chunk.
// This is returned when Stream is true.
type ChatStreamResponse struct {
	ID string `json:"id"`
	// Unix timestamp in seconds.
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	SystemFingerprint string             `json:"system_fingerprint"`
	Choices           []ChatStreamChoice `json:"choices"`
	Usage             *ChatUsage         `json:"usage,omitempty"`
}

// ChatStreamChoice is a single streaming completion choice.
type ChatStreamChoice struct {
	// Required.
	Delta        ChatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason,omitempty"`
	// Required.
	Index    int           `json:"index"`
	LogProbs *ChatLogProbs `json:"logprobs,omitempty"`
}

// ChatStreamDelta holds incremental updates for a stream.
// Deltas may include content/role/tool_call_id or role/tool_calls.
type ChatStreamDelta struct {
	// Content is present for text deltas.
	Content *string `json:"content,omitempty"`
	// Role may be included with the first delta.
	Role *string `json:"role,omitempty"`
	// ToolCallID may be included for tool-specific content.
	ToolCallID *string `json:"tool_call_id,omitempty"`
	// ToolCalls is present for tool call deltas.
	ToolCalls []ChatStreamToolCall `json:"tool_calls,omitempty"`
}

// UnmarshalJSON enforces the union shape for ChatStreamDelta.
func (d *ChatStreamDelta) UnmarshalJSON(data []byte) error {
	type alias ChatStreamDelta
	var tmp alias
	if err := json.Unmarshal(data, &tmp); err != nil {
		return err
	}
	out := ChatStreamDelta(tmp)
	if err := out.validate(); err != nil {
		return err
	}
	*d = out

	return nil
}

// validate enforces the union shape for ChatStreamDelta.
func (d ChatStreamDelta) validate() error {
	if len(d.ToolCalls) > 0 {
		if d.Content != nil || d.ToolCallID != nil {
			return &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindValidation,
				Message: "stream delta: tool_calls cannot include content or tool_call_id",
				Err:     nil,
			}
		}
	}

	return nil
}

// ChatStreamToolCall represents a tool call within a streaming delta.
type ChatStreamToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Index    int                `json:"index"`
	Function ChatStreamFunction `json:"function"`
}

// ChatStreamFunction represents a streamed function call.
type ChatStreamFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
