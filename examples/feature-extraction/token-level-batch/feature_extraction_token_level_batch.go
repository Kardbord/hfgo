package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Create a new client with your API token and a transformers-style model.
	// Unlike sentence-transformers models, this returns one vector per token.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("facebook/bart-base"),
	)

	inputs := []string{
		"Transformers is an awesome library!",
		"Token-level embeddings are returned for every token.",
	}

	fmt.Println("Extracting token-level features from inputs:")
	PrintJSON(inputs)
	fmt.Println("...")

	// Make the batched token-level feature extraction request
	embeddings, err := client.ExtractTokenFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: inputs,
		},
	)
	if err != nil {
		log.Fatalf("error extracting batched token-level features: %v\n", err)
	}

	fmt.Printf("Results: %d inputs\n", len(embeddings))
	for inputIndex, tokens := range embeddings {
		if len(tokens) == 0 {
			fmt.Printf("  input %d: no tokens\n", inputIndex)

			continue
		}
		fmt.Printf(
			"  input %d: %d tokens, hidden size %d\n",
			inputIndex,
			len(tokens),
			len(tokens[0]),
		)
	}
	PrintJSON(embeddings)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
