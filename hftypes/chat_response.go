package hftypes

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ChatResponse represents a non-streaming completion response from the chat API.
// This is returned when Stream is false (the default).
//
// Decoding is intentionally tolerant of provider variations. Use [ChatResponse.Validate]
// to enforce the strict Hugging Face chat-completion schema.
type ChatResponse struct {
	ID string `json:"id"`

	// Unix timestamp in seconds.
	Created int64 `json:"created"`

	Model             string       `json:"model"`
	SystemFingerprint string       `json:"system_fingerprint"`
	Choices           []ChatChoice `json:"choices"`
	Usage             ChatUsage    `json:"usage"`
}

// ChatChoice is a single non-streaming completion choice.
type ChatChoice struct {
	FinishReason string                `json:"finish_reason"`
	Index        int                   `json:"index"`
	LogProbs     *ChatLogProbs         `json:"logprobs,omitempty"`
	Message      ChatCompletionMessage `json:"message"`
}

// ChatLogProbs contains per-token log probabilities.
type ChatLogProbs struct {
	Content []ChatLogProb `json:"content"`
}

// ChatLogProb contains logprob information for a token.
type ChatLogProb struct {
	Token       string           `json:"token"`
	LogProb     float64          `json:"logprob"`
	TopLogProbs []ChatTopLogProb `json:"top_logprobs"`
}

// ChatTopLogProb is a top log probability entry for a token position.
type ChatTopLogProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
}

// ChatCompletionMessage is a message returned by the model.
// It may contain text content, tool calls, or both.
//
// Decoding accepts provider-specific variations. Use [ChatCompletionMessage.Validate]
// to enforce the Hugging Face schema's strict text-or-tool-call union.
type ChatCompletionMessage struct {
	Role string `json:"role"`
	// Content is present for text responses.
	Content *string `json:"content,omitempty"`
	// ToolCallID is set when returning tool-specific content.
	ToolCallID *string `json:"tool_call_id,omitempty"`
	// ToolCalls is present for tool call responses.
	ToolCalls []ChatToolCallOutput `json:"tool_calls,omitempty"`
}

// ChatToolCallOutput represents a tool call in a response message.
type ChatToolCallOutput struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ChatFunctionCall `json:"function"`
}

// Validate enforces the Hugging Face chat-completion schema on the response.
// It returns an [hferrors.SDKError] with [hferrors.SDKErrorKindValidation] when
// a choice does not contain exactly one of text content or tool calls, or when
// tool-call metadata is invalid.
//
// Validation is opt-in because some inference providers return OpenAI-compatible
// responses that do not follow the strict schema. Decoding does not call Validate.
func (r ChatResponse) Validate() error {
	if len(r.Choices) == 0 {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindValidation,
			Message: "completion response: no choices",
			Err:     nil,
		}
	}
	for i := range r.Choices {
		if err := r.Choices[i].Message.Validate(); err != nil {
			return fmt.Errorf("choice %d: %w", i, err)
		}
	}

	return nil
}

// Validate enforces the Hugging Face text-or-tool-call union on the message.
// It returns an [hferrors.SDKError] with [hferrors.SDKErrorKindValidation] when
// the message does not match either variant.
//
// Validation is opt-in because some inference providers return OpenAI-compatible
// responses where content and tool calls coexist. Decoding does not call Validate.
func (m ChatCompletionMessage) Validate() error {
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
	for i := range m.ToolCalls {
		if err := m.ToolCalls[i].validate(); err != nil {
			return err
		}
	}

	return nil
}

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

	return c.Function.validate()
}

// ChatUsage contains token usage statistics.
type ChatUsage struct {
	CompletionTokens int `json:"completion_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
