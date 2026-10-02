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

	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("dslim/bert-base-NER"),
	)

	inputs := []string{
		"My name is Sarah and I live in London.",
		"I work at Google in New York.",
		"Angela Merkel was the Chancellor of Germany.",
	}

	fmt.Println("Classifying tokens in batch:")
	PrintJSON(inputs)
	fmt.Println("...")

	results, err := client.ClassifyTokensBatch(
		hftypes.TokenClassificationBatchRequest{
			Inputs: inputs,
		},
	)
	if err != nil {
		log.Fatalf("error running batch token classification: %v\n", err)
	}

	fmt.Println("Results:")
	for i, entities := range results {
		fmt.Printf("\nInput %d: %q\n", i, inputs[i])
		PrintJSON(entities)
	}
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
