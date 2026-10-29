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
	"github.com/stretchr/testify/require"
)

// newOVHcloudClient returns a client configured with the OVHcloud provider, so
// chat requests are routed to OVHcloud through the Hugging Face router.
// Callers gate the test on HF_TOKEN.
func newOVHcloudClient(ctx context.Context) hfgo.Client {
	return hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel("openai/gpt-oss-120b"),
		hfopts.WithProvider(hfproviders.NewOVHcloudProvider()),
		hfopts.WithContext(ctx),
	)
}

// TestChatCompletion_OVHcloudLiveAPI tests a basic chat completion routed to
// OVHcloud through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_OVHcloudLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newOVHcloudClient(ctx)

	const text = "Say hello in one sentence."
	resp, err := client.Chat(
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

	require.NoError(t, err, "ovhcloud chat completion should succeed")
	require.NotEmpty(t, resp.ID, "response should have an ID")
	require.NotEmpty(t, resp.Model, "response should have a model")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	choice := resp.Choices[0]
	require.Equal(t, "assistant", choice.Message.Role, "message should have the assistant role")
	require.NotNil(t, choice.Message.Content, "message should have content")
	require.NotEmpty(t, *choice.Message.Content, "message content should not be empty")
	require.NotEmpty(t, choice.FinishReason, "choice should have a finish reason")
}

// TestChatCompletion_OVHcloudStreamingLiveAPI tests streaming chat completion
// routed to OVHcloud through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_OVHcloudStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newOVHcloudClient(ctx)

	text := "Say hello in one sentence."
	stream, err := client.ChatStream(
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
	require.NoError(t, err, "streaming request should succeed")
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

// TestChatCompletion_OVHcloudMultiMessageLiveAPI tests a multi-turn
// conversation routed to OVHcloud through the live HF API. This test requires
// the HF_TOKEN environment variable to be set.
func TestChatCompletion_OVHcloudMultiMessageLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newOVHcloudClient(ctx)

	text1 := "What is 2+2?"
	text2 := "What is the answer multiplied by 3?"

	resp, err := client.Chat(
		hftypes.ChatRequest{
			Messages: []hftypes.ChatMessage{
				{
					Role: "user",
					Content: hftypes.ChatMessageContent{
						Text: &text1,
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
						Text: &text2,
					},
				},
			},
		},
	)

	require.NoError(t, err, "multi-message chat completion should succeed")
	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotNil(t, resp.Choices[0].Message.Content, "message should have content")
	require.NotEmpty(t, *resp.Choices[0].Message.Content, "message content should not be empty")
}
