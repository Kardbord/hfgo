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
		hfopts.WithModel("meta-llama/Llama-3.1-8B-Instruct"),
	)

	// ProviderParams is the escape hatch for OpenAI-compatible chat body
	// keys the typed fields do not cover. Entries merge into the request
	// body's top level, overriding typed fields at the same level, and the
	// resolved model cannot be overridden. See
	// docs/providers/huggingface.md for the full semantics.
	providerParams := map[string]any{
		"logit_bias": map[string]int{"1120": -100},
	}

	fmt.Println("Sending a chat request with a ProviderParams entry:")
	PrintJSON(providerParams)
	fmt.Println("...")

	response, err := client.Chat(
		hftypes.ChatRequest{
			Messages: []hftypes.ChatMessage{{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: new("Write a haiku about the ocean."),
				},
			}},
			ProviderParams: providerParams,
		},
	)
	if err != nil {
		log.Fatalf("error running chat: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(response)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
