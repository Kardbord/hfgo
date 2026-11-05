//go:build integration

package integration_tests

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// newScalewayClient returns a client configured with the Scaleway provider
// backed by a Hub model resolver, so pipeline tasks resolve Scaleway's model
// identifiers from the Hub's inferenceProviderMapping data. Callers gate the
// test on HF_TOKEN.
func newScalewayClient(t *testing.T, ctx context.Context, model string) hfgo.Client {
	t.Helper()

	provider, err := hfproviders.NewScalewayProvider(
		hfproviders.NewHubModelResolver(
			hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
		),
	)
	require.NoError(t, err)

	return hfgo.NewClient(
		hfopts.WithToken(os.Getenv("HF_TOKEN")),
		hfopts.WithModel(model),
		hfopts.WithProvider(provider),
		hfopts.WithContext(ctx),
	)
}

// TestChatCompletion_ScalewayLiveAPI tests a basic chat completion routed to
// Scaleway through the live HF API. This test requires the HF_TOKEN
// environment variable to be set.
func TestChatCompletion_ScalewayLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newScalewayClient(t, ctx, "openai/gpt-oss-120b")

	const text = "Say hello in one sentence."
	resp := testutils.RetryTransient(ctx, t, func() (hftypes.ChatResponse, error) {
		return client.Chat(
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
	require.NotEmpty(t, choice.FinishReason, "finish reason should not be empty")
}

// TestFeatureExtractionPooled_ScalewayLiveAPI tests a pooled feature
// extraction request routed to Scaleway through the live HF API. This test
// requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooled_ScalewayLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := newScalewayClient(t, ctx, "Qwen/Qwen3-Embedding-8B")

	resp := testutils.RetryTransient(
		ctx,
		t,
		func() (hftypes.FeatureExtractionPooledEmbedding, error) {
			return client.ExtractPooledFeatures(
				hftypes.FeatureExtractionRequest{
					Input: "What is the capital of France?",
				},
			)
		},
	)

	require.NotEmpty(t, resp, "embedding should not be empty")
	require.Positive(t, resp.Dimension(), "embedding should have a positive dimension")
}

// TestFeatureExtractionPooledBatch_ScalewayLiveAPI tests batch pooled
// feature extraction routed to Scaleway through the live HF API. This test
// requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooledBatch_ScalewayLiveAPI(t *testing.T) {
	require.NotEmpty(t, os.Getenv("HF_TOKEN"), "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := newScalewayClient(t, ctx, "Qwen/Qwen3-Embedding-8B")

	inputs := []string{
		"The weather is nice today.",
		"I love programming in Go.",
	}

	resp := testutils.RetryTransient(
		ctx,
		t,
		func() ([]hftypes.FeatureExtractionPooledEmbedding, error) {
			return client.ExtractPooledFeaturesBatch(
				hftypes.FeatureExtractionBatchRequest{
					Inputs: inputs,
				},
			)
		},
	)

	require.Len(t, resp, 2, "response should have 2 embeddings")

	for i, embedding := range resp {
		require.NotEmpty(t, embedding, "embedding %d should not be empty", i)
		require.Positive(
			t,
			embedding.Dimension(),
			"embedding %d should have a positive dimension", i,
		)
	}
}
