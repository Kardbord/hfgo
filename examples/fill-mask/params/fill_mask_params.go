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
		hfopts.WithModel("google-bert/bert-base-uncased"),
	)

	input := "The quick brown fox jumps over the [MASK] dog."

	targets := []string{
		"lazy",
		"smart",
		"fast",
	}

	fmt.Println("Filling the mask in input:")
	PrintJSON(input)
	fmt.Println("Restricting predictions to targets:")
	PrintJSON(targets)
	fmt.Println("...")

	// Make the fill mask request with parameters
	predictions, err := client.FillMask(
		hftypes.FillMaskRequest{
			Input: input,
			Parameters: &hftypes.FillMaskParameters{
				TopK:    new(5),
				Targets: targets,
			},
		},
	)
	if err != nil {
		log.Fatalf("error running fill mask: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(predictions)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
