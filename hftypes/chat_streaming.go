package hftypes

import (
	"context"
	"fmt"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ChatStreamResponse represents a single streaming response chunk. It is the
// DTO returned by [ChatStream.Recv]; [ChatStream] itself is the stream handle.
// This is returned when Stream is true.
//
// Decoding is intentionally tolerant of provider variations. Use
// [ChatStreamResponse.Validate] to enforce the Hugging Face streaming schema.
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
	Delta        ChatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason,omitempty"`
	Index        int             `json:"index"`
	LogProbs     *ChatLogProbs   `json:"logprobs,omitempty"`
}

// ChatStreamDelta holds incremental updates for a stream.
// Deltas may include content/role/tool_call_id or role/tool_calls.
//
// Decoding accepts provider-specific variations. Use [ChatStreamDelta.Validate]
// to enforce the Hugging Face schema's strict text-or-tool-call union.
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

// Validate enforces the Hugging Face streaming union on all choices in the chunk.
// It returns an [hferrors.SDKError] with [hferrors.SDKErrorKindValidation] when
// a delta mixes text content with tool calls.
//
// Validation is opt-in because some inference providers return OpenAI-compatible
// deltas where content and tool calls coexist. Decoding does not call Validate.
func (r ChatStreamResponse) Validate() error {
	// Empty choices are valid: providers may send usage-only or terminal
	// chunks without a choice delta.
	for i := range r.Choices {
		if err := r.Choices[i].Delta.Validate(); err != nil {
			return fmt.Errorf("choice %d: %w", i, err)
		}
	}

	return nil
}

// Validate enforces the Hugging Face text-or-tool-call union on the delta.
// It returns an [hferrors.SDKError] with [hferrors.SDKErrorKindValidation] when
// tool calls are mixed with content or tool_call_id.
//
// Streaming deltas are incremental, so this validates only the union shape, not
// every tool-call field. Validation is opt-in because some inference providers
// return OpenAI-compatible deltas where content and tool calls coexist.
func (d ChatStreamDelta) Validate() error {
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

// toolCallAccumulator tracks the last known metadata for a streamed tool call.
// It fills in missing id/type/function-name fields across incremental deltas so
// the consumer always observes complete tool-call identifiers. It is owned by a
// single consumer and is not safe for concurrent use.
type toolCallAccumulator struct {
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

// merge records any provided metadata for a tool call and returns the
// accumulated values for that (choice, call) pair.
func (a *toolCallAccumulator) merge(
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
func (a *toolCallAccumulator) state(choiceIndex, callIndex int) *toolCallState {
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

// ChatStream is a handle for consuming a streaming chat completion response.
// Each [ChatStream.Recv] call returns a [ChatStreamResponse] chunk; ChatStream
// is not itself a response DTO.
//
// A ChatStream is not safe for concurrent Recv calls. Close may be called
// concurrently with Recv. To fan out chunks, distribute them from a single
// consumer loop.
type ChatStream struct {
	stream       chatstreamer
	toolCallAccr toolCallAccumulator
}

// NewChatStream wraps a JSON chat completion stream with tool call accumulation.
func NewChatStream(stream chatstreamer) *ChatStream {
	return &ChatStream{
		stream:       stream,
		toolCallAccr: toolCallAccumulator{cache: nil},
	}
}

// Recv blocks until the next streaming chunk arrives or the context is done.
// It returns io.EOF when the stream ends. Termination is final: once io.EOF
// is observed, no further chunks will be delivered, and every delivered
// chunk precedes the stream's terminal frame.
//
// Recv is not safe to call concurrently on the same ChatStream.
func (c *ChatStream) Recv(ctx context.Context) (ChatStreamResponse, error) {
	if c == nil || c.stream == nil {
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

// Close releases the underlying stream resources. It may be called
// concurrently with Recv.
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
			toolID, callType, functionName := c.toolCallAccr.merge(
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
