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
		hfopts.WithModel("facebook/bart-large-mnli"),
	)

	inputs := []string{
		"This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three.",
		"This could have been better. The director was completely unfaithful to the books, and it moved at a snail's pace.",
	}

	candidates := []string{
		"enthralled",
		"bored",
	}

	fmt.Println("Classifying inputs:")
	PrintJSON(inputs)
	fmt.Println("Into candidate labels:")
	PrintJSON(candidates)
	fmt.Println("...")

	// Make the classification request
	classifications, err := client.ZeroShotClassifyTextBatch(
		hftypes.ZeroShotTextClassificationBatchRequest{
			Inputs: inputs,
			Parameters: &hftypes.ZeroShotTextClassificationParameters{
				CandidateLabels: candidates,
				MultiLabel:      Ptr(false),
			},
		},
	)
	if err != nil {
		log.Fatalf("error running text classification: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(classifications)
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
