package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// This example demonstrates how to use the client for streaming chat completions.
// It sends a message to the model and processes the response stream in real-time.
func main() {
	// Get the API token from environment variable
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Create a new client with your API token and model
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("deepseek-ai/DeepSeek-R1"),
	)

	// Create a chat request for streaming
	prompt := "Tell me a short joke about programming."
	request := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: &prompt,
				},
			},
		},
		StreamOptions: &hftypes.ChatStreamOptions{
			IncludeUsage: new(false),
		},
		MaxTokens: new(1024),
	}

	// Create a context for the streaming request/response
	ctx := context.Background()

	// Send the streaming request
	stream, err := client.ChatStream(request, hfopts.WithContext(ctx))
	if err != nil {
		log.Fatalf("Failed to start streaming chat request: %v", err)
	}

	// Remember to close the stream when done to release resources
	defer stream.Close()

	fmt.Printf("Prompt: %s\n\n", prompt)
	fmt.Println("Streaming Chat Completion Response:")
	fmt.Println("-----------------------------------")

	// Process the stream chunks as they arrive
	for {
		chunk, err := stream.Recv(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			log.Printf("Error receiving stream chunk: %v", err)

			return
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		// Process the chunk
		choice := chunk.Choices[0]
		if choice.Delta.Content != nil {
			fmt.Print(*choice.Delta.Content)
		}
	}

	fmt.Println("\n-----------------------------------")
	fmt.Println("Stream completed successfully!")
}
