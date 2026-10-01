package chatstream

import (
	"context"

	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// ChatStream wraps a streaming chat completion response.
type ChatStream struct {
	stream       *request.JSONStream[dto.ChatStreamResponse]
	toolCallAccr ToolCallAccumulator
}

// NewChatStream wraps a JSON chat completion stream with tool call accumulation.
func NewChatStream(stream *request.JSONStream[dto.ChatStreamResponse]) *ChatStream {
	return &ChatStream{
		stream:       stream,
		toolCallAccr: ToolCallAccumulator{cache: nil},
	}
}

// Recv blocks until the next streaming chunk arrives or the context is done.
func (c *ChatStream) Recv(ctx context.Context) (dto.ChatStreamResponse, error) {
	if c.stream == nil {
		return dto.ChatStreamResponse{}, &hferrors.SDKError{
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
func (c *ChatStream) mergeToolCallMetadata(resp *dto.ChatStreamResponse) {
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
