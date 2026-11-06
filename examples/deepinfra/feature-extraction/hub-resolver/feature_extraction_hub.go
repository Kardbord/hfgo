package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Build a Hub model resolver, which queries the Hub's
	// inferenceProviderMapping data for the configured model ID instead of
	// using a pinned table. The token raises rate limits and is required for
	// gated models; results are cached per resolver instance.
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(token),
	)

	// Create a DeepInfra provider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewDeepInfraProvider(resolver)
	if err != nil {
		log.Fatalf("error creating DeepInfra provider: %v", err)
	}

	// Configure the client to route feature-extraction requests through the
	// DeepInfra provider. The mapped model rides the request body, so the
	// hf-inference parameters the canonical request carries (e.g. normalize)
	// are not forwarded; DeepInfra-specific parameters such as
	// encoding_format and dimensions ride the request's ProviderParams field
	// instead.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("BAAI/bge-base-en-v1.5"),
		hfopts.WithProvider(provider),
	)

	input := "The quick brown fox jumps over the lazy dog."

	fmt.Println("Extracting features from input:")
	PrintJSON(input)
	fmt.Println("...")

	// Make the feature extraction request
	embedding, err := client.ExtractPooledFeatures(
		context.Background(),
		hftypes.FeatureExtractionRequest{
			Input: input,
		},
	)
	if err != nil {
		log.Fatalf("error extracting features: %v\n", err)
	}

	fmt.Printf("Results: embedding with %d dimensions\n", embedding.Dimension())
	PrintJSON(embedding)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
