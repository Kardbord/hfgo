package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// FalAIProvider serves text-to-image through the Hugging Face router using
// fal-ai's own model identifiers. Callers configure the Hugging Face model
// ID they know; the provider translates it to fal-ai's identifier at
// endpoint-construction time via the [hfproviders.ModelResolver] it was
// constructed with. The Hugging Face router is a path-transparent proxy, so
// the endpoint path embeds the provider slug and the provider's own model
// ID. Chat needs no such mapping: the router resolves chat models
// server-side.
type FalAIProvider struct {
	hfproviders.HuggingFaceTextToImageMixin

	mapper hfproviders.ModelResolver
}

// NewFalAIProvider creates a [FalAIProvider] backed by the given model
// resolver, e.g. a [hfproviders.HubModelResolver] for live Hub mappings or
// a [hfproviders.StaticModelResolver] for pinned deployments.
func NewFalAIProvider(mapper hfproviders.ModelResolver) FalAIProvider {
	if mapper == nil {
		panic("NewFalAIProvider: mapper must not be nil")
	}

	return FalAIProvider{mapper: mapper}
}

func (FalAIProvider) Name() string           { return "fal-ai" }
func (FalAIProvider) ProviderSuffix() string { return "" }

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// fal-ai's identifier and returns the router-proxied path for it. The
// provider slug given to the resolver is fal-ai's Hub identifier — the key
// the Hub's mapping data uses — which is a provider fact, not something the
// SDK derives. The wire-format codec comes from the mixin; a partner whose
// payload shape differs from the Hugging Face format would supply its own
// codec the same way.
func (p FalAIProvider) TextToImageEndpoint(
	params hfproviders.EndpointParams,
) (hfproviders.Endpoint, error) {
	providerID, err := p.mapper.ResolveModel(params.Context, hfproviders.ResolveModelParams{
		Provider: "fal-ai",
		Task:     "text-to-image",
		Model:    params.Model,
	})
	if err != nil {
		return hfproviders.Endpoint{}, err
	}

	return hfproviders.Endpoint{Method: http.MethodPost, Path: "fal-ai/" + providerID}, nil
}

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// The Hub-backed resolver performs network I/O only when an endpoint
	// method asks for a mapping; constructing it has no side effects.
	mapper := hfproviders.NewHubModelResolver(hfproviders.WithHubToken(token))

	// Generated images are binary and can exceed the 1 MiB default response
	// cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithProvider(NewFalAIProvider(mapper)),
		hfopts.WithModel("black-forest-labs/FLUX.1-dev"),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "A serene mountain landscape at sunset"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: prompt,
	})
	if err != nil {
		log.Fatalf("error generating image: %v\n", err)
	}

	fmt.Printf("Received %d bytes of %s\n", len(image.Image), image.MediaType)
}
