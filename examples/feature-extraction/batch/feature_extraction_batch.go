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

	inputs := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is transforming software engineering.",
		"Go makes concurrency straightforward.",
	}

	fmt.Println("Extracting features from inputs:")
	PrintJSON(inputs)
	fmt.Println("...")

	// Make the batched feature extraction request
	embeddings, err := client.ExtractFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: inputs,
		},
	)
	if err != nil {
		log.Fatalf("error extracting batched features: %v\n", err)
	}

	fmt.Printf("Results: %d embeddings\n", len(embeddings))
	for i, emb := range embeddings {
		fmt.Printf("  embedding %d: %d dimensions\n", i, len(emb))
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
