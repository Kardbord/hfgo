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
		hfopts.WithModel("facebook/bart-large-cnn"),
	)

	inputs := []string{
		"The Earth orbits the Sun once every 365 days, giving us a year.",
		"The Moon orbits the Earth roughly every 27 days, a period known as a sidereal month.",
	}

	fmt.Println("Summarizing inputs:")
	PrintJSON(inputs)
	fmt.Println("...")

	// Make the batched summarization request
	summaries, err := client.SummarizeBatch(
		hftypes.SummarizationBatchRequest{
			Inputs: inputs,
		},
	)
	if err != nil {
		log.Fatalf("error running batched summarization: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(summaries)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
