package hfproviders_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// testClient returns a factory that produces an HTTP client with a dedicated
// transport. This keeps the example isolated from environment proxy settings
// and pointed at the local httptest.Server.
func testClient() func() http.Client {
	return func() http.Client {
		//nolint:exhaustruct // Zero-value Transport uses no proxy and HTTP/1.1.
		return http.Client{Transport: &http.Transport{}}
	}
}

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

// Example_customProvider demonstrates registering a custom provider with a
// client. Implement [hfproviders.Provider] plus the per-task interfaces you
// support, pass the value to [hfopts.WithProvider], and use the client
// normally: the SDK casts the provider and routes matching calls to it.
func Example_customProvider() {
	// The response body is hard-coded so the example can run without a real token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This provider routes text classification to a ":echo" path.
		if r.URL.Path != "/models/my-model:echo" {
			w.WriteHeader(http.StatusNotFound)

			return
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `[{"label":"positive","score":0.95}]`)
	}))
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("my-model"),
		hfopts.WithProvider(EchoProvider{}),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.ClassifyText(hftypes.TextClassificationRequest{
		Input: "I love this product!",
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (%.2f)\n", result[0].Label, result[0].Score)
	// Output: positive (0.95)
}
