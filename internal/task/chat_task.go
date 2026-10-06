package task

import (
	"fmt"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// resolveModel resolves the model with precedence and applies the provider suffix.
// Model precedence: request > options.
// Provider suffix is appended only if the model doesn't already contain a provider
// (indicated by ":") and the provider is not the default HuggingFace provider.
// Returns the resolved model along with its suffix, or an error if no model could be resolved.
func resolveModel(payload *hftypes.ChatRequest, opts hfopts.Options) (string, error) {
	if payload.Model == nil || *payload.Model == "" {
		if opts.Model != "" {
			model := opts.Model
			payload.Model = &model
		}
	}
	payload.Model = applyProvider(payload.Model, opts.Provider)

	if payload.Model == nil || *payload.Model == "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "no chat model specified",
			Err:     nil,
		}
	}

	return *payload.Model, nil
}

// applyProvider applies the provider to the model if the model
// doesn't already contain a provider (indicated by ":").
func applyProvider(model *string, provider hfproviders.Provider) *string {
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

// Chat sends a chat completion request and returns a chat completion response.
//
//nolint:gocritic // hugeParam: Chat takes the request by value so the SDK never mutates the caller's payload
func Chat(opts hfopts.Options, req hftypes.ChatRequest) (hftypes.ChatResponse, error) {
	model, err := resolveModel(&req, opts)
	if err != nil {
		return hftypes.ChatResponse{}, err
	}
	opts = opts.With(hfopts.WithModel(model))

	if err = validateDispatch(opts); err != nil {
		return hftypes.ChatResponse{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.ChatProvider](opts.Provider)
	if err != nil {
		return hftypes.ChatResponse{}, err
	}

	endpoint, err := prov.ChatEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   model,
	})
	if err != nil {
		return hftypes.ChatResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing chat endpoint: " + err.Error(),
			Err:     err,
		}
	}

	if req.Stream != nil && *req.Stream {
		return hftypes.ChatResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "use a streaming chat method instead",
			Err:     nil,
		}
	}

	return doInference(
		opts,
		endpoint,
		prov.ChatCodec(),
		req,
	)
}

// StreamChat sends a chat completion request and returns a streaming response.
//
//nolint:gocritic // hugeParam: StreamChat takes the request by value so the SDK never mutates the caller's payload
func StreamChat(opts hfopts.Options, req hftypes.ChatRequest) (*hftypes.ChatStream, error) {
	model, err := resolveModel(&req, opts)
	if err != nil {
		return nil, err
	}
	opts = opts.With(hfopts.WithModel(model))

	if err = validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.ChatStreamProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.ChatEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing chat streaming endpoint: " + err.Error(),
			Err:     err,
		}
	}

	stream := true
	req.Stream = &stream

	streamResp, err := doStreamingInference(
		opts,
		endpoint,
		prov.ChatStreamCodec(),
		req,
	)
	if err != nil {
		return nil, err
	}

	return hftypes.NewChatStream(streamResp), nil
}
