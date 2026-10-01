//go:build integration

package task

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// TestChatCompletion_LiveAPI tests a basic chat completion against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestChatCompletion_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Use a model known to be available on HF Inference API
	const model = "deepseek-ai/DeepSeek-R1"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	const text = "Say hello in one sentence."
	resp, err := client.Chat(
		ChatRequest{
			Messages: []ChatMessage{
				{
					Role: "user",
					Content: ChatMessageContent{
						Text: testutils.Ptr(text),
					},
				},
			},
		},
	)

	require.NoError(t, err, "chat completion should succeed")
	require.NotEmpty(t, resp.ID, "response should have an ID")
	require.NotEmpty(t, resp.Model, "response should have a model")
	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotEmpty(t, resp.Choices, "should have at least one choice")

	choice := resp.Choices[0]
	require.NotNil(t, choice.Message, "choice should have a message")
	require.NotEmpty(t, choice.Message.Role, "message should have a role")
	require.NotNil(t, choice.Message.Content, "message should have content")
	require.NotEmpty(t, *choice.Message.Content, "message content should not be empty")
	require.NotEmpty(t, choice.FinishReason, "choice should have a finish reason")

	// Verify usage statistics are present
	require.Positive(t, resp.Usage.PromptTokens, "should have prompt tokens")
	require.Positive(t, resp.Usage.CompletionTokens, "should have completion tokens")
	require.Positive(t, resp.Usage.TotalTokens, "should have total tokens")
	require.Equal(t,
		resp.Usage.PromptTokens+resp.Usage.CompletionTokens,
		resp.Usage.TotalTokens,
		"total tokens should equal sum of prompt and completion tokens",
	)
}

// TestChatCompletion_StreamingLiveAPI tests streaming chat completion against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestChatCompletion_StreamingLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Use a model known to be available on HF Inference API
	const model = "deepseek-ai/DeepSeek-R1"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	text := "Say hello in one sentence."
	stream, err := client.ChatStream(
		ChatRequest{
			Messages: []ChatMessage{
				{
					Role: "user",
					Content: ChatMessageContent{
						Text: &text,
					},
				},
			},
		},
	)
	require.NoError(t, err, "streaming request should succeed")
	defer stream.Close()

	// Collect all chunks
	var chunkCount int
	var totalContent strings.Builder

	for {
		chunk, err := stream.Recv(ctx)
		if err != nil {
			require.ErrorIs(t, err, io.EOF, "recv error must be EOF")

			break
		}

		chunkCount++
		require.NotEmpty(t, chunk.ID, "chunk should have an ID")

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.Delta.Content != nil {
				totalContent.WriteString(*choice.Delta.Content)
			}
		}
	}

	require.Positive(t, chunkCount, "should receive at least one chunk")
	require.NotEmpty(t, totalContent, "should accumulate content from chunks")
}

// TestChatCompletion_MultiMessageLiveAPI tests a multi-turn conversation against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestChatCompletion_MultiMessageLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Use a model known to be available on HF Inference API
	const model = "deepseek-ai/DeepSeek-R1"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	text1 := "What is 2+2?"
	text2 := "What is the answer multiplied by 3?"

	resp, err := client.Chat(
		ChatRequest{
			Messages: []ChatMessage{
				{
					Role: "user",
					Content: ChatMessageContent{
						Text: &text1,
					},
				},
				{
					Role: "assistant",
					Content: ChatMessageContent{
						Text: testutils.Ptr("2+2 equals 4"),
					},
				},
				{
					Role: "user",
					Content: ChatMessageContent{
						Text: &text2,
					},
				},
			},
		},
	)

	require.NoError(t, err, "multi-message chat completion should succeed")
	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotEmpty(t, resp.Choices, "should have at least one choice")
	require.NotNil(t, resp.Choices[0].Message.Content, "message should have content")
	require.NotEmpty(t, *resp.Choices[0].Message.Content, "message content should not be empty")
}
