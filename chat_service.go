package hfgo

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/internal/chatstream"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// chatService implements chat completion calls using the configured request options.
type chatService struct {
	opts request.Options
}

// newChatService builds a chat service with a snapshot of the provided options.
func newChatService(opts request.Options) chatService {
	return chatService{opts: opts}
}

// resolveModel resolves the model with precedence and applies the provider suffix.
// Model precedence: request > options > client.
// Provider suffix is appended only if the model doesn't already contain a provider
// (indicated by ":") and the provider is not the default HuggingFace provider.
func resolveModel(payload *ChatRequest, optsOverride request.Options) {
	// Resolve model with precedence: request > options > client
	if payload.Model == nil || *payload.Model == "" {
		if optsOverride.Model != "" {
			model := optsOverride.Model
			payload.Model = &model
		}
	}

	// Apply provider fallback to the final model
	payload.Model = applyProvider(payload.Model, optsOverride.Provider)
}

// applyProvider applies the provider to the model if the model
// doesn't already contain a provider (indicated by ":").
func applyProvider(model *string, provider Provider) *string {
	if model == nil || *model == "" || provider == nil {
		return model
	}

	if provider.ProviderSuffix() == "" {
		return model
	}

	if !strings.Contains(*model, ":") {
		newModel := fmt.Sprintf("%s:%s", *model, provider.ProviderSuffix())

		return &newModel
	}

	return model
}

// resolveEndpoint resolves the chat completion endpoint, validating the model
// and provider. It returns the endpoint path and the resolved options override.
func (s chatService) resolveEndpoint(
	req *ChatRequest,
	opts ...Option,
) (string, request.Options, error) {
	optsOverride := s.opts.With(opts...)

	resolveModel(req, optsOverride)

	if req.Model == nil || *req.Model == "" {
		return "", request.Options{}, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "the model option must be set for chat completion to succeed",
			Err:     nil,
		}
	}

	provider := optsOverride.Provider
	if provider == nil {
		return "", request.Options{}, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "provider must not be nil",
			Err:     nil,
		}
	}

	endpoint, err := provider.Endpoint(TaskChatCompletion, *req.Model)
	if err != nil {
		return "", request.Options{}, err
	}

	return endpoint, optsOverride, nil
}

// complete sends a chat completion request and returns a chat completion response.
//
//nolint:gocritic // hugeParam: complete takes the request by value so the SDK never mutates the caller's payload
func (s chatService) complete(req ChatRequest, opts ...Option) (ChatResponse, error) {
	endpoint, optsOverride, err := s.resolveEndpoint(&req, opts...)
	if err != nil {
		return ChatResponse{}, err
	}

	if req.Stream != nil && *req.Stream {
		return ChatResponse{}, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "chat completion streaming is not supported; use a streaming chat method instead",
			Err:     nil,
		}
	}

	return request.DoJSON[ChatRequest, ChatResponse](
		optsOverride,
		http.MethodPost,
		endpoint,
		req,
	)
}

// completeStream sends a chat completion request and returns a streaming response.
//
//nolint:gocritic // hugeParam: completeStream takes the request by value so the SDK never mutates the caller's payload
func (s chatService) completeStream(req ChatRequest, opts ...Option) (*ChatStream, error) {
	endpoint, optsOverride, err := s.resolveEndpoint(&req, opts...)
	if err != nil {
		return nil, err
	}

	stream := true
	req.Stream = &stream

	streamResp, err := request.DoJSONStream[ChatRequest, ChatStreamResponse](
		optsOverride,
		http.MethodPost,
		endpoint,
		req,
	)
	if err != nil {
		return nil, err
	}

	return &ChatStream{
		stream:       streamResp,
		toolCallAccr: chatstream.ToolCallAccumulator{},
	}, nil
}

// ChatStream wraps a streaming chat completion response.
type ChatStream struct {
	stream       *request.JSONStream[ChatStreamResponse]
	toolCallAccr chatstream.ToolCallAccumulator
}

// Recv blocks until the next streaming chunk arrives or the context is done.
func (c *ChatStream) Recv(ctx context.Context) (ChatStreamResponse, error) {
	if c.stream == nil {
		return ChatStreamResponse{}, &SDKError{
			Kind:    SDKErrorKindInternal,
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
