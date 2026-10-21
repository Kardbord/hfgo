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

	input := "Transformers is an awesome library!"

	fmt.Println("Extracting token-level features from input:")
	PrintJSON(input)
	fmt.Println("...")

	// Make the token-level feature extraction request
	tokens, err := client.ExtractTokenFeatures(
		hftypes.FeatureExtractionRequest{
			Input: input,
		},
	)
	if err != nil {
		log.Fatalf("error extracting token-level features: %v\n", err)
	}

	fmt.Printf("Results: %d tokens\n", len(tokens))
	if len(tokens) > 0 {
		fmt.Printf("  hidden size: %d\n", len(tokens[0]))
	}
	PrintJSON(tokens)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
