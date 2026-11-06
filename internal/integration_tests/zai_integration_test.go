//go:build integration

package integration_tests

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// newZAIClient returns a client configured with the Z.ai provider, so chat
// requests are routed to Z.ai through the Hugging Face router. Callers gate
// the test on HF_TOKEN.
func newZAIClient(ctx context.Context) hfgo.Client {
	return hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel("zai-org/GLM-5.3"),
		hfopts.WithProvider(hfproviders.NewZAIProvider()),
	)
}

// TestChatCompletion_ZAILiveAPI tests a basic chat completion routed to Z.ai
// through the live HF API. This test requires the HF_TOKEN environment
// variable to be set.
func TestChatCompletion_ZAILiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newZAIClient(ctx)

	const text = "Say hello in one sentence."
	resp := testutils.RetryTransient(ctx, t, func() (hftypes.ChatResponse, error) {
		return client.Chat(ctx,
			hftypes.ChatRequest{
				Messages: []hftypes.ChatMessage{
					{
						Role: "user",
						Content: hftypes.ChatMessageContent{
							Text: new(text),
						},
					},
				},
			},
		)
	})

	require.NotEmpty(t, resp.ID, "response should have an ID")
	require.NotEmpty(t, resp.ID, "response should have an ID")
	require.NotEmpty(t, resp.Model, "response should have a model")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	choice := resp.Choices[0]
	require.Equal(t, "assistant", choice.Message.Role, "message should have the assistant role")
	require.NotNil(t, choice.Message.Content, "message should have content")
	require.NotEmpty(t, *choice.Message.Content, "message content should not be empty")
	require.NotEmpty(t, choice.FinishReason, "choice should have a finish reason")
}

// TestChatCompletion_ZAIStreamingLiveAPI tests streaming chat completion
// routed to Z.ai through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_ZAIStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newZAIClient(ctx)

	text := "Say hello in one sentence."
	stream := testutils.RetryTransient(ctx, t, func() (*hftypes.ChatStream, error) {
		return client.ChatStream(ctx,
			hftypes.ChatRequest{
				Messages: []hftypes.ChatMessage{
					{
						Role: "user",
						Content: hftypes.ChatMessageContent{
							Text: &text,
						},
					},
				},
			},
		)
	})
	defer stream.Close()

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

// TestChatCompletion_ZAIMultiMessageLiveAPI tests a multi-turn conversation
// routed to Z.ai through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_ZAIMultiMessageLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newZAIClient(ctx)

	const text1 = "What is 2+2?"
	const text2 = "What is the answer multiplied by 3?"

	resp := testutils.RetryTransient(ctx, t, func() (hftypes.ChatResponse, error) {
		return client.Chat(ctx,
			hftypes.ChatRequest{
				Messages: []hftypes.ChatMessage{
					{
						Role: "user",
						Content: hftypes.ChatMessageContent{
							Text: new(text1),
						},
					},
					{
						Role: "assistant",
						Content: hftypes.ChatMessageContent{
							Text: new("2+2 equals 4"),
						},
					},
					{
						Role: "user",
						Content: hftypes.ChatMessageContent{
							Text: new(text2),
						},
					},
				},
			},
		)
	})

	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotNil(t, resp.Choices[0].Message.Content, "message should have content")
	require.NotEmpty(t, *resp.Choices[0].Message.Content, "message content should not be empty")
}
