package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const Model = "deepseek-ai/DeepSeek-R1"

func main() {
	chatClient := NewChatClient(
		"You are a friendly chat bot whose purpose is to chat banally with users. Always respond in no more than two brief sentences.",
	)
	if err := chatClient.Chat(); err != nil {
		log.Fatalf("ChatClient encountered an error: %v\n", err)
	}
}

type ChatClient struct {
	hfClient hfgo.Client
	scanner  *bufio.Scanner
	history  []hftypes.ChatMessage
}

func NewChatClient(sysPrompt string) ChatClient {
	const tokenEnv = "HF_TOKEN"
	token := os.Getenv(tokenEnv)
	if token == "" {
		log.Fatalf("%s environment variable is not set\n", tokenEnv)
	}

	chatClient := ChatClient{
		hfClient: hfgo.NewClient(
			hfopts.WithToken(token),
			hfopts.WithModel(Model),
		),
		scanner: bufio.NewScanner(os.Stdin),
		history: make([]hftypes.ChatMessage, 1, 20),
	}

	chatClient.history[0] = hftypes.ChatMessage{
		Role: "system",
		Content: hftypes.ChatMessageContent{
			Text: &sysPrompt,
		},
	}

	return chatClient
}

func (chatClient *ChatClient) Chat() error {
	fmt.Println("Welcome to this chat bot example! Press Ctrl+d at any time to exit.")
	fmt.Println("Initializing...")
	stream, err := chatClient.hfClient.ChatStream(
		hftypes.ChatRequest{
			Messages: chatClient.history, // Initialize with the system prompt given to NewChatClient
		},
	)
	if err != nil {
		_ = stream.Close()

		return err
	}

	err = chatClient.recv(stream)
	_ = stream.Close()
	if err != nil {
		return err
	}

	fmt.Print("\n> ")
	for chatClient.scanner.Scan() {
		input := strings.TrimSpace(chatClient.scanner.Text())
		if input == "" {
			fmt.Print("\n> ")

			continue
		}

		stream, err := chatClient.prompt(input)
		if err != nil {
			_ = stream.Close()

			return err
		}

		err = chatClient.recv(stream)
		_ = stream.Close()
		if err != nil {
			return err
		}

		fmt.Print("\n> ")
	}
	fmt.Println()

	return chatClient.scanner.Err()
}

func (chatClient *ChatClient) prompt(input string) (*hftypes.ChatStream, error) {
	chatClient.history = append(chatClient.history, hftypes.ChatMessage{
		Role: "user",
		Content: hftypes.ChatMessageContent{
			Text: &input,
		},
	})

	return chatClient.hfClient.ChatStream(
		hftypes.ChatRequest{
			Messages: chatClient.history,
		},
		hfopts.WithContext(context.Background()),
	)
}

func (chatClient *ChatClient) recv(stream *hftypes.ChatStream) error {
	response := strings.Builder{}
	for {
		chunk, err := stream.Recv(context.Background())
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}

			return err
		}

		chunkContent := ""
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != nil {
			chunkContent = *chunk.Choices[0].Delta.Content
		}
		response.WriteString(chunkContent)
		fmt.Print(chunkContent)
	}

	chatClient.history = append(chatClient.history, hftypes.ChatMessage{
		Role: "assistant",
		Content: hftypes.ChatMessageContent{
			Text: new(response.String()),
		},
	})

	return nil
}
