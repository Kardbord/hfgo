package hftypes

import (
	"context"
	"encoding/json"

	"github.com/Kardbord/hfgo/v4/hferrors"
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

// ToolCallAccumulator tracks the last known metadata for a streamed tool call.
// It fills in missing id/type/function-name fields across incremental deltas so
// callers always observe complete tool-call identifiers.
type ToolCallAccumulator struct {
	cache map[key]*toolCallState
}

// key identifies a tool call within a streaming response choice.
type key struct {
	choice int
	call   int
}

// toolCallState holds the sticky metadata captured for a tool call index.
type toolCallState struct {
	ID           string
	Type         string
	FunctionName string
}

// Merge records any provided metadata for a tool call and returns the
// accumulated values for that (choice, call) pair.
func (a *ToolCallAccumulator) Merge(
	choiceIndex int,
	callIndex int,
	toolID string,
	toolType string,
	functionName string,
) (finalID, finalType, finalName string) {
	state := a.state(choiceIndex, callIndex)
	if toolID != "" {
		state.ID = toolID
	}
	if toolType != "" {
		state.Type = toolType
	}
	if functionName != "" {
		state.FunctionName = functionName
	}

	return state.ID, state.Type, state.FunctionName
}

// state returns the cached metadata for the provided tool call, creating a new
// entry when this (choice, call) pair is observed for the first time.
func (a *ToolCallAccumulator) state(choiceIndex, callIndex int) *toolCallState {
	if a.cache == nil {
		a.cache = make(map[key]*toolCallState)
	}
	cacheKey := key{choice: choiceIndex, call: callIndex}
	if state, ok := a.cache[cacheKey]; ok {
		return state
	}
	state := &toolCallState{
		ID:           "",
		Type:         "",
		FunctionName: "",
	}
	a.cache[cacheKey] = state

	return state
}

type chatstreamer interface {
	Recv(ctx context.Context) (ChatStreamResponse, error)
	Close() error
}

// ChatStream wraps a streaming chat completion response.
type ChatStream struct {
	stream       chatstreamer
	toolCallAccr ToolCallAccumulator
}

// NewChatStream wraps a JSON chat completion stream with tool call accumulation.
func NewChatStream(stream chatstreamer) *ChatStream {
	return &ChatStream{
		stream:       stream,
		toolCallAccr: ToolCallAccumulator{cache: nil},
	}
}

// Recv blocks until the next streaming chunk arrives or the context is done.
func (c *ChatStream) Recv(ctx context.Context) (ChatStreamResponse, error) {
	if c.stream == nil {
		return ChatStreamResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "chat stream is nil",
			Err:     nil,
		}
	}

	chunk, err := c.stream.Recv(ctx)
	if err != nil {
		return chunk, err
	}

	c.mergeToolCallMetadata(&chunk)

	return chunk, nil
}

// Close releases the underlying stream resources.
func (c *ChatStream) Close() error {
	if c == nil || c.stream == nil {
		return nil
	}

	return c.stream.Close()
}

// mergeToolCallMetadata ensures streaming tool call deltas include the cached
// id/type/function-name values observed earlier in the stream.
func (c *ChatStream) mergeToolCallMetadata(resp *ChatStreamResponse) {
	if c == nil || resp == nil {
		return
	}
	for i := range resp.Choices {
		choice := &resp.Choices[i]
		if len(choice.Delta.ToolCalls) == 0 {
			continue
		}
		for j := range choice.Delta.ToolCalls {
			call := &choice.Delta.ToolCalls[j]
			toolID, callType, functionName := c.toolCallAccr.Merge(
				choice.Index,
				call.Index,
				call.ID,
				call.Type,
				call.Function.Name,
			)
			call.ID = toolID
			call.Type = callType
			call.Function.Name = functionName
		}
	}
}
