//go:build integration

package integration_tests

import (
	"context"
	"io"
	"net/http"
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

const (
	// nscaleChatModel is a conversational model Nscale serves on the router.
	nscaleChatModel = "zai-org/GLM-5.3"
	// nscaleTextToImageModel is a text-to-image model Nscale serves on the
	// router.
	nscaleTextToImageModel = "black-forest-labs/FLUX.1-schnell"
)

// newNscaleProvider returns an [hfproviders.NscaleProvider] backed by a Hub
// model resolver for live API tests. Callers gate the test on HF_TOKEN.
func newNscaleProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
	)

	provider, err := hfproviders.NewNscaleProvider(resolver)
	require.NoError(t, err)

	return provider
}

// TestChatCompletion_NscaleLiveAPI tests a basic chat completion routed to
// Nscale through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_NscaleLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel(nscaleChatModel),
		hfopts.WithProvider(newNscaleProvider(t)),
	)

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
	require.NotEmpty(t, resp.Model, "response should have a model")
	require.NotEmpty(t, resp.Choices, "response should have choices")

	choice := resp.Choices[0]
	require.Equal(t, "assistant", choice.Message.Role, "message should have the assistant role")
	require.NotNil(t, choice.Message.Content, "message should have content")
	require.NotEmpty(t, *choice.Message.Content, "message content should not be empty")
	require.NotEmpty(t, choice.FinishReason, "choice should have a finish reason")
}

// TestChatCompletion_NscaleStreamingLiveAPI tests streaming chat completion
// routed to Nscale through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_NscaleStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel(nscaleChatModel),
		hfopts.WithProvider(newNscaleProvider(t)),
	)

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

// TestGenerateImage_NscaleLiveAPI tests a basic text-to-image request against
// the live HF API using the Nscale provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_NscaleLiveAPI(t *testing.T) {
	t.Skip("skipped until Nscale hosts a model that supports the text to image task on huggingface")
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(nscaleTextToImageModel),
		hfopts.WithProvider(newNscaleProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	response := testutils.RetryTransient(ctx, t, func() (hftypes.TextToImageResponse, error) {
		return client.GenerateImage(ctx, hftypes.TextToImageRequest{
			Input: "A serene mountain landscape at sunset",
			Parameters: &hftypes.TextToImageParameters{
				Width:  new(512),
				Height: new(512),
			},
		})
	})

	require.NotEmpty(t, response.Image, "generated image should not be empty")
	require.NotEmpty(t, response.Image, "generated image should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "image/"),
		"expected an image media type, got %s",
		response.MediaType,
	)

	imageType := http.DetectContentType(response.Image)
	require.True(
		t,
		strings.HasPrefix(imageType, "image/"),
		"expected image data, got %s",
		imageType,
	)
}
