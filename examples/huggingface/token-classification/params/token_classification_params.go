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

	input := "My name is Sarah and I live in London."

	fmt.Println("Classifying tokens with aggregation strategy:")
	PrintJSON(input)
	fmt.Println("...")

	entities, err := client.ClassifyTokens(
		hftypes.TokenClassificationRequest{
			Input: input,
			Parameters: &hftypes.TokenClassificationParameters{
				AggregationStrategy: new(hftypes.TokenClassificationAggregationSimple),
				IgnoreLabels:        []string{"O"},
			},
		},
	)
	if err != nil {
		log.Fatalf("error running token classification: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(entities)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
