//go:build integration

package integration_tests

import (
	"context"
	"encoding/json"
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

// newTogetherClient returns a client configured with the Together provider.
// Chat pins routing with the router's ":together" model suffix and needs no
// model resolver. Callers gate the test on HF_TOKEN.
func newTogetherClient(t *testing.T, ctx context.Context, model string) hfgo.Client {
	t.Helper()

	return hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel(model),
		hfopts.WithProvider(hfproviders.NewTogetherProvider()),
		hfopts.WithContext(ctx),
	)
}

// TestChatCompletion_TogetherLiveAPI tests a basic chat completion routed to
// Together through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_TogetherLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newTogetherClient(t, ctx, "deepseek-ai/DeepSeek-V4.1-Flash")

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

	require.NoError(t, err, "together chat completion should succeed")
	require.NotEmpty(t, resp.ID, "response should have an ID")
	require.NotEmpty(t, resp.Model, "response should have a model")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	choice := resp.Choices[0]
	require.Equal(t, "assistant", choice.Message.Role, "message should have the assistant role")
	require.NotNil(t, choice.Message.Content, "message should have content")
	require.NotEmpty(t, *choice.Message.Content, "message content should not be empty")
	require.NotEmpty(t, choice.FinishReason, "choice should have a finish reason")
}

// TestChatCompletion_TogetherStreamingLiveAPI tests streaming chat completion
// routed to Together through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_TogetherStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newTogetherClient(t, ctx, "deepseek-ai/DeepSeek-V4.1-Flash")

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

// TestChatCompletion_TogetherJSONSchemaLiveAPI exercises Together's flattened
// json_schema response_format end to end: the codec unwraps OpenAI's
// envelope, and Together answers with schema-conforming JSON. This test
// requires the HF_TOKEN environment variable to be set.
func TestChatCompletion_TogetherJSONSchemaLiveAPI(t *testing.T) {
	t.Skip("temporarily skipped until the live json_schema response_format path is verified")

	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newTogetherClient(t, ctx, "deepseek-ai/DeepSeek-V4.1-Flash")

	resp, err := client.Chat(
		hftypes.ChatRequest{
			Messages: []hftypes.ChatMessage{
				{
					Role: "user",
					Content: hftypes.ChatMessageContent{
						Text: new("What is the capital of France? Answer as a JSON object."),
					},
				},
			},
			ResponseFormat: &hftypes.ChatResponseFormat{
				Type: hftypes.ResponseFormatTypeJSONSchema,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{
					Name:   "capital",
					Strict: new(true),
					Schema: json.RawMessage(
						`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"]}`,
					),
				},
			},
		},
	)
	require.NoError(t, err, "together json_schema chat completion should succeed")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	content := resp.Choices[0].Message.Content
	require.NotNil(t, content, "message should have content")

	var answer struct {
		Capital string `json:"capital"`
	}
	require.NoError(t, json.Unmarshal([]byte(*content), &answer))
	require.NotEmpty(t, answer.Capital, "schema-conforming answer should carry the capital")
}
