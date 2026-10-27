package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hfproviders"
)

const fooProviderName = "foo"

// FooProvider serves text-to-image through the Hugging Face router using a
// fictional "foo" partner's own model identifiers. Callers configure the
// Hugging Face model ID they know; the provider translates it to foo's
// identifier at endpoint-construction time via the [hfproviders.ModelResolver]
// it was constructed with. The Hugging Face router is a path-transparent
// proxy, so the endpoint path embeds the provider slug and the provider's own
// model ID. Chat needs no such mapping: the router resolves chat models
// server-side.
type FooProvider struct {
	hfproviders.HuggingFaceTextToImageMixin

	mapper hfproviders.ModelResolver
}

// Compile-time assertion that FooProvider serves text-to-image.
var _ hfproviders.TextToImageProvider = FooProvider{}

// NewFooProvider creates a [FooProvider] backed by the given model resolver,
// e.g. a [hfproviders.HubModelResolver] for live Hub mappings or a
// [hfproviders.StaticModelResolver] for pinned deployments.
func NewFooProvider(mapper hfproviders.ModelResolver) FooProvider {
	if mapper == nil {
		panic("NewFooProvider: mapper must not be nil")
	}

	return FooProvider{mapper: mapper}
}

func (FooProvider) Name() string           { return fooProviderName }
func (FooProvider) ProviderSuffix() string { return "" }

// TextToImageEndpoint resolves the configured Hugging Face model ID to foo's
// identifier and returns the router-proxied path for it. The provider slug
// given to the resolver is foo's Hub identifier — the key the Hub's mapping
// data uses — which is a provider fact, not something the SDK derives. The
// wire-format codec comes from the mixin; a partner whose payload shape
// differs from the Hugging Face format would supply its own codec the same way.
func (p FooProvider) TextToImageEndpoint(
	params hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	providerModelID, err := p.mapper.ResolveModel(params.Context, hfproviders.ResolveModelParams{
		Provider: fooProviderName,
		Task:     hfproviders.TaskTextToImage,
		Model:    params.Model,
	})
	if err != nil {
		return hfproviders.Endpoint{}, err
	}

	return hfproviders.Endpoint{Method: http.MethodPost, Path: "foo/" + providerModelID}, nil
}

func main() {
	// A static resolver keeps the example self-contained: it performs no
	// network I/O and needs no token. Swap in
	// hfproviders.NewHubModelResolver(hfproviders.WithHubToken(...)) to resolve
	// live Hub mappings instead.
	mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]string{
		{
			Provider: fooProviderName,
			Task:     hfproviders.TaskTextToImage,
			Model:    "black-forest-labs/FLUX.1-dev",
		}: "foo/flux-dev",
	})

	provider := NewFooProvider(mapper)

	endpoint, err := provider.TextToImageEndpoint(hfproviders.EndpointParams{
		Context: context.Background(),
		Model:   "black-forest-labs/FLUX.1-dev",
	})
	if err != nil {
		log.Fatalf("error resolving endpoint: %v", err)
	}

	fmt.Printf("foo text-to-image endpoint: %s %s\n", endpoint.Method, endpoint.Path)
}
