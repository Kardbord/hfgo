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

	input := "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. " +
		"Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel " +
		"Tower surpassed the Washington Monument to become the tallest man-made structure in the world."

	fmt.Println("Summarizing input:")
	PrintJSON(input)
	fmt.Println("...")

	// Make the summarization request
	summaries, err := client.Summarize(
		hftypes.SummarizationRequest{
			Input: input,
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
