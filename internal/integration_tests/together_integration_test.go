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
	"github.com/Kardbord/hfgo/v4/internal/testutils"
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

// TestChatCompletion_TogetherStreamingLiveAPI tests streaming chat completion
// routed to Together through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_TogetherStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newTogetherClient(t, ctx, "deepseek-ai/DeepSeek-V4.1-Flash")

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

// TestChatCompletion_TogetherJSONSchemaLiveAPI exercises Together's
// structured-output path end to end: the request leaves the SDK in OpenAI's
// nested json_schema response_format envelope — the shape Together's current
// API requires — and Together answers with schema-conforming JSON. This test
// requires the HF_TOKEN environment variable to be set.
//
// Skipped pending an upstream fix: the Hugging Face router's Together
// mapping rewrites response_format before forwarding, flattening OpenAI's
// nested envelope to {"type":"json_schema","schema":...} — a shape
// Together's current API rejects with 400 json_parse_error ("missing field
// `json_schema`"). A direct request to api.together.xyz with the nested
// envelope succeeds, so the SDK-side encoding is correct (pinned by
// TestTogetherProvider_ResponseFormatPassthrough and
// TestClientChat_TogetherProvider); no client-side change can make the
// routed path pass. Re-enable once the router forwards the body untouched.
func TestChatCompletion_TogetherJSONSchemaLiveAPI(t *testing.T) {
	t.Skip("skipped until the Hugging Face router stops flattening " +
		"json_schema response formats, which Together's current API rejects")
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newTogetherClient(t, ctx, "deepseek-ai/DeepSeek-V4.1-Flash")

	resp := testutils.RetryTransient(ctx, t, func() (hftypes.ChatResponse, error) {
		return client.Chat(ctx,
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
	})
	require.NotEmpty(t, resp.Choices, "response should have choices")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	content := resp.Choices[0].Message.Content
	require.NotNil(t, content, "message should have content")

	var answer struct {
		Capital string `json:"capital"`
	}
	require.NoError(t, json.Unmarshal([]byte(*content), &answer))
	require.NotEmpty(t, answer.Capital, "schema-conforming answer should carry the capital")
}
