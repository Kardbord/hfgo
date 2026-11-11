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

// newDeepInfraClient returns a client configured with the DeepInfra provider
// backed by a Hub model resolver, so pipeline tasks resolve DeepInfra's model
// identifiers from the Hub's inferenceProviderMapping data. Callers gate the
// test on HF_TOKEN.
func newDeepInfraClient(t *testing.T, ctx context.Context, model string) hfgo.Client {
	t.Helper()

	provider, err := hfproviders.NewDeepInfraProvider(
		hfproviders.NewHubModelResolver(
			hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
		),
	)
	require.NoError(t, err)

	return hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel(model),
		hfopts.WithProvider(provider),
	)
}

// TestChatCompletion_DeepInfraLiveAPI tests a basic chat completion routed to
// DeepInfra through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_DeepInfraLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newDeepInfraClient(t, ctx, "openai/gpt-oss-120b")

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

// TestChatCompletion_DeepInfraStreamingLiveAPI tests streaming chat
// completion routed to DeepInfra through the live HF API. This test requires
// the HF_TOKEN environment variable to be set.
func TestChatCompletion_DeepInfraStreamingLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newDeepInfraClient(t, ctx, "openai/gpt-oss-120b")

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

// TestRecognizeSpeech_DeepInfraLiveAPI tests a speech recognition request
// routed to DeepInfra through the live HF API, exercising the multipart
// transcription exchange end to end. This test requires the HF_TOKEN
// environment variable to be set.
func TestRecognizeSpeech_DeepInfraLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := newDeepInfraClient(t, ctx, "openai/whisper-large-v3")

	resp := testutils.RetryTransient(ctx, t, func() (hftypes.SpeechRecognitionResponse, error) {
		return client.RecognizeSpeech(ctx, hftypes.SpeechRecognitionRequest{
			Input: speechRecognitionInput(t),
		})
	})

	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
}

// TestFeatureExtractionPooled_DeepInfraLiveAPI tests a pooled feature
// extraction request routed to DeepInfra through the live HF API, exercising
// the OpenAI-style embeddings exchange end to end. This test requires the
// HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooled_DeepInfraLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newDeepInfraClient(t, ctx, "BAAI/bge-base-en-v1.5")

	resp := testutils.RetryTransient(
		ctx,
		t,
		func() (hftypes.FeatureExtractionPooledEmbedding, error) {
			return client.ExtractPooledFeatures(ctx,
				hftypes.FeatureExtractionRequest{
					Input: "What is the capital of France?",
				},
			)
		},
	)

	require.NotEmpty(t, resp, "embedding should not be empty")
	require.Greater(t, resp.Dimension(), 0, "embedding should have a positive dimension")
}

// TestFeatureExtractionPooledBatch_DeepInfraLiveAPI tests batch pooled
// feature extraction routed to DeepInfra through the live HF API. This test
// requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooledBatch_DeepInfraLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := newDeepInfraClient(t, ctx, "BAAI/bge-base-en-v1.5")

	inputs := []string{
		"The weather is nice today.",
		"I love programming in Go.",
	}

	resp := testutils.RetryTransient(
		ctx,
		t,
		func() ([]hftypes.FeatureExtractionPooledEmbedding, error) {
			return client.ExtractPooledFeaturesBatch(ctx,
				hftypes.FeatureExtractionBatchRequest{
					Inputs: inputs,
				},
			)
		},
	)

	require.Len(t, resp, 2, "response should have 2 embeddings")

	for i, embedding := range resp {
		require.NotEmpty(t, embedding, "embedding %d should not be empty", i)
		require.Greater(
			t,
			embedding.Dimension(),
			0,
			"embedding %d should have a positive dimension", i,
		)
	}
}
