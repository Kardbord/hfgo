package task

import (
	"fmt"
	"strings"

	"github.com/Kardbord/hfgo/v4/internal/chatstream"
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

// resolveModel resolves the model with precedence and applies the provider suffix.
// Model precedence: request > options.
// Provider suffix is appended only if the model doesn't already contain a provider
// (indicated by ":") and the provider is not the default HuggingFace provider.
func resolveModel(payload *dto.ChatRequest, opts request.Options) {
	if payload.Model == nil || *payload.Model == "" {
		if opts.Model != "" {
			model := opts.Model
			payload.Model = &model
		}
	}

	payload.Model = applyProvider(payload.Model, opts.Provider)
}

// applyProvider applies the provider to the model if the model
// doesn't already contain a provider (indicated by ":").
func applyProvider(model *string, provider providers.Provider) *string {
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

// resolveChatOptions validates options and resolves the model on the request payload.
// It returns the options with the model set so downstream dispatch can use it.
func resolveChatOptions(opts request.Options, req *dto.ChatRequest) (request.Options, error) {
	resolveModel(req, opts)

	if req.Model == nil || *req.Model == "" {
		return request.Options{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "the model option must be set for chat completion to succeed",
			Err:     nil,
		}
	}

	if opts.Provider == nil {
		return request.Options{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "provider must not be nil",
			Err:     nil,
		}
	}

	opts.Model = *req.Model

	return opts, nil
}

// Chat sends a chat completion request and returns a chat completion response.
//
//nolint:gocritic // hugeParam: Chat takes the request by value so the SDK never mutates the caller's payload
func Chat(opts request.Options, req dto.ChatRequest) (dto.ChatResponse, error) {
	optsOverride, err := resolveChatOptions(opts, &req)
	if err != nil {
		return dto.ChatResponse{}, err
	}

	if req.Stream != nil && *req.Stream {
		return dto.ChatResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "chat completion streaming is not supported; use a streaming chat method instead",
			Err:     nil,
		}
	}

	return doJSONInference[dto.ChatRequest, dto.ChatResponse](
		optsOverride,
		providers.TaskChatCompletion,
		req,
	)
}

// StreamChat sends a chat completion request and returns a streaming response.
//
//nolint:gocritic // hugeParam: DoChatStream takes the request by value so the SDK never mutates the caller's payload
func StreamChat(opts request.Options, req dto.ChatRequest) (*chatstream.ChatStream, error) {
	optsOverride, err := resolveChatOptions(opts, &req)
	if err != nil {
		return nil, err
	}

	stream := true
	req.Stream = &stream

	streamResp, err := doStreamingInference[dto.ChatRequest, dto.ChatStreamResponse](
		optsOverride,
		providers.TaskChatCompletion,
		req,
	)
	if err != nil {
		return nil, err
	}

	return chatstream.NewChatStream(streamResp), nil
}
