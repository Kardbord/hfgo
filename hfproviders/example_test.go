package hfproviders_test

import (
	"context"
	"fmt"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// EchoProvider is a minimal custom provider. It prepends a provider suffix to
// model IDs for routing and reuses the built-in JSON codec for every task it
// supports. This is for illustration only; a real third-party provider would
// translate requests and responses to its own wire format.
type EchoProvider struct{}

// ProviderSuffix returns the routing token appended to model IDs for tasks that
// carry the model in the request body.
func (EchoProvider) ProviderSuffix() string { return "echo" }

// Name returns the provider name used in error messages and debugging.
func (EchoProvider) Name() string { return "echo-provider" }

// TextClassificationEndpoint returns the endpoint for text classification.
func (EchoProvider) TextClassificationEndpoint(
	params hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{
		Method: http.MethodPost,
		Path:   "models/" + params.Model + ":echo",
	}, nil
}

// TextClassificationCodec returns the JSON codec for text classification.
func (EchoProvider) TextClassificationCodec() hfproviders.TextClassificationCodec {
	return hfproviders.JSONCodec[hftypes.TextClassificationRequest, []hftypes.TextClassification]{}
}

// SummarizationEndpoint returns the endpoint for summarization.
func (EchoProvider) SummarizationEndpoint(
	params hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	return hfproviders.Endpoint{
		Method: http.MethodPost,
		Path:   "models/" + params.Model + ":echo/summarize",
	}, nil
}

// SummarizationCodec returns the JSON codec for summarization.
func (EchoProvider) SummarizationCodec() hfproviders.SummarizationCodec {
	return hfproviders.JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{}
}

// ExampleAsProvider demonstrates type-safe casting of a provider to a
// per-task interface.
func ExampleAsProvider() {
	p := hfproviders.NewHuggingFaceProvider()

	chatProvider, err := hfproviders.AsProvider[hfproviders.ChatProvider](p)
	if err != nil {
		panic(err)
	}

	endpoint, err := chatProvider.ChatEndpoint(hfproviders.EndpointParams{
		Context: context.Background(),
		Model:   "test-model",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(endpoint.Path)
	// Output: v1/chat/completions
}

// ExampleEchoProvider demonstrates implementing a custom provider by hand and
// casting it to the task-specific interface consumed by the typed dispatch
// layer.
func ExampleEchoProvider() {
	// Any value that implements Provider plus the relevant per-task
	// interfaces can be supplied to hfgo.NewClient via hfopts.WithProvider.
	custom := EchoProvider{}

	tcProvider, err := hfproviders.AsProvider[hfproviders.TextClassificationProvider](custom)
	if err != nil {
		panic(err)
	}

	endpoint, err := tcProvider.TextClassificationEndpoint(hfproviders.EndpointParams{
		Context: context.Background(),
		Model:   "my-model",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(endpoint.Path)
	// Output: models/my-model:echo
}
