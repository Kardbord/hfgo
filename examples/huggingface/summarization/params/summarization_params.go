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

	input := "The industrial revolution transformed agriculture, manufacturing, mining, and transport, " +
		"leading to massive social and economic changes across the world."

	cleanUp := true
	truncation := hftypes.SummarizationTruncationOnlyFirst

	fmt.Println("Summarizing input:")
	PrintJSON(input)
	fmt.Println("...")

	// Make the summarization request with custom parameters
	summaries, err := client.Summarize(
		hftypes.SummarizationRequest{
			Input: input,
			Parameters: &hftypes.SummarizationParameters{
				CleanUpTokenizationSpaces: &cleanUp,
				Truncation:                &truncation,
			},
		},
	)
	if err != nil {
		log.Fatalf("error running summarization: %v\n", err)
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
