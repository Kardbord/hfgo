package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// This example demonstrates how to use the client for basic (non-streaming)
// chat completions. It sends a single message to the model and prints the response.
func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Create a new client with your API token and desired model
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("deepseek-ai/DeepSeek-R1"),
	)

	// Create a chat request with a simple message
	request := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: new("Hello! What is the capital of France?"),
				},
			},
		},
		MaxTokens: new(1024),
	}

	// Send the request and get the response
	response, err := client.Chat(request)
	if err != nil {
		log.Fatalf("Failed to complete chat request: %v", err)
	}

	// Print the response
	fmt.Println("Chat Completion Response:")
	fmt.Printf("  ID: %s\n", response.ID)
	fmt.Printf("  Model: %s\n", response.Model)
	fmt.Printf("  Choices: %d\n", len(response.Choices))

	for i, choice := range response.Choices {
		fmt.Printf("\n  Choice %d:\n", i)
		fmt.Printf("    Finish Reason: %s\n", choice.FinishReason)
		fmt.Printf("    Role: %s\n", choice.Message.Role)
		if choice.Message.Content != nil {
			fmt.Printf("    Content: %s\n", *choice.Message.Content)
		}
	}

	fmt.Printf("\nUsage:\n")
	fmt.Printf("  Prompt Tokens: %d\n", response.Usage.PromptTokens)
	fmt.Printf("  Completion Tokens: %d\n", response.Usage.CompletionTokens)
	fmt.Printf("  Total Tokens: %d\n", response.Usage.TotalTokens)
}
