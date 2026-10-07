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

	// Create a new client with your API token and desired model
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("sentence-transformers/all-MiniLM-L6-v2"),
	)

	input := "The quick brown fox jumps over the lazy dog."
	params := hftypes.FeatureExtractionParameters{
		Normalize: Ptr(true),
		Truncate:  Ptr(true),
	}

	fmt.Println("Extracting features from input:")
	PrintJSON(input)
	fmt.Println("With parameters:")
	PrintJSON(params)
	fmt.Println("...")

	// Make the feature extraction request with parameters
	embedding, err := client.FeatureExtract(
		hftypes.FeatureExtractionRequest{
			Input:      input,
			Parameters: &params,
		},
	)
	if err != nil {
		log.Fatalf("error extracting features: %v\n", err)
	}

	fmt.Printf("Results: embedding with %d dimensions\n", len(embedding))
	PrintJSON(embedding)
}

func Ptr[T any](v T) *T {
	return &v
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
