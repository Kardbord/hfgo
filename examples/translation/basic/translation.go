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
		hfopts.WithModel("google-t5/t5-small"),
	)

	input := "Hello, how are you doing today?"

	fmt.Println("Translating input:")
	PrintJSON(input)
	fmt.Println("...")

	// Make the translation request with source and target languages
	translations, err := client.Translate(
		hftypes.TranslationRequest{
			Input: input,
		},
	)
	if err != nil {
		log.Fatalf("error running translation: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(translations)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
