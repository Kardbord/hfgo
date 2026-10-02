package hftypes

import (
	"encoding/json"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ChatResponse represents a non-streaming completion response from the chat API.
// This is returned when Stream is false (the default).
type ChatResponse struct {
	// Required.
	ID string `json:"id"`
	// Unix timestamp in seconds.
	// Required.
	Created int64 `json:"created"`
	// Required.
	Model string `json:"model"`
	// Required.
	SystemFingerprint string `json:"system_fingerprint"`
	// Required.
	Choices []ChatChoice `json:"choices"`
	// Required.
	Usage ChatUsage `json:"usage"`
}

// ChatChoice is a single non-streaming completion choice.
type ChatChoice struct {
	// Required.
	FinishReason string `json:"finish_reason"`
	// Required.
	Index    int           `json:"index"`
	LogProbs *ChatLogProbs `json:"logprobs,omitempty"`
	// Required.
	Message ChatCompletionMessage `json:"message"`
}

// ChatLogProbs contains per-token log probabilities.
type ChatLogProbs struct {
	// Required.
	Content []ChatLogProb `json:"content"`
}

// ChatLogProb contains logprob information for a token.
type ChatLogProb struct {
	// Required.
	Token string `json:"token"`
	// Required.
	LogProb float64 `json:"logprob"`
	// Required.
	TopLogProbs []ChatTopLogProb `json:"top_logprobs"`
}

// ChatTopLogProb is a top log probability entry for a token position.
type ChatTopLogProb struct {
	// Required.
	Token string `json:"token"`
	// Required.
	LogProb float64 `json:"logprob"`
}

// ChatCompletionMessage is a message returned by the model.
// It is either a text message (Content) or a tool call message (ToolCalls).
type ChatCompletionMessage struct {
	Role string `json:"role"`
	// Content is present for text responses.
	Content *string `json:"content,omitempty"`
	// ToolCallID is set when returning tool-specific content.
	ToolCallID *string `json:"tool_call_id,omitempty"`
	// ToolCalls is present for tool call responses.
	ToolCalls []ChatToolCallOutput `json:"tool_calls,omitempty"`
}

// UnmarshalJSON enforces the union shape for ChatCompletionMessage.
func (m *ChatCompletionMessage) UnmarshalJSON(data []byte) error {
	type alias ChatCompletionMessage
	var tmp alias
	if err := json.Unmarshal(data, &tmp); err != nil {
		return err
	}
	out := ChatCompletionMessage(tmp)
	if err := out.validate(); err != nil {
		return err
	}
	*m = out

	return nil
}

// validate enforces the union shape for ChatCompletionMessage.
func (m ChatCompletionMessage) validate() error {
	if m.Content != nil {
		if len(m.ToolCalls) > 0 {
			return &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindValidation,
				Message: "completion message: content and tool_calls are mutually exclusive",
				Err:     nil,
			}
		}
	} else if len(m.ToolCalls) == 0 {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "completion message: either content or tool_calls must be set",
			Err:     nil,
		}
	}
	if m.Content == nil && m.ToolCallID != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "completion message: tool_call_id requires content",
			Err:     nil,
		}
	}

	return nil
}

// ChatToolCallOutput represents a tool call in a response message.
type ChatToolCallOutput struct {
	// Required.
	ID string `json:"id"`
	// Required.
	Type string `json:"type"`
	// Required.
	Function ChatFunctionCall `json:"function"`
}

// UnmarshalJSON enforces the tool call shape for ChatToolCallOutput.
func (c *ChatToolCallOutput) UnmarshalJSON(data []byte) error {
	type alias ChatToolCallOutput
	var tmp alias
	if err := json.Unmarshal(data, &tmp); err != nil {
		return err
	}
	out := ChatToolCallOutput(tmp)
	if err := out.validate(); err != nil {
		return err
	}
	*c = out

	return nil
}

// validate enforces the tool call shape for ChatToolCallOutput.
func (c ChatToolCallOutput) validate() error {
	if c.ID == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "chat tool call output: id must be set",
			Err:     nil,
		}
	}
	if c.Type == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "chat tool call output: type must be set",
			Err:     nil,
		}
	}

	return nil
}

// ChatUsage contains token usage statistics.
type ChatUsage struct {
	// Required.
	CompletionTokens int `json:"completion_tokens"`
	// Required.
	PromptTokens int `json:"prompt_tokens"`
	// Required.
	TotalTokens int `json:"total_tokens"`
}
