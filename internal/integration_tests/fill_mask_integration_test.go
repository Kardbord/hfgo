//go:build integration

package hfgo

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestFillMask_LiveAPI tests a basic fill mask request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-bert/bert-base-uncased"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	resp, err := client.FillMask(
		FillMaskRequest{
			Input: "The capital of France is [MASK].",
		},
	)

	require.NoError(t, err, "fill mask should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have predictions")

	for _, prediction := range resp {
		require.NotEmpty(t, prediction.Sequence, "prediction should have a sequence")
		require.GreaterOrEqual(t, prediction.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, prediction.Score, 1.0, "score should be at most 1.0")
		require.NotZero(t, prediction.Token, "prediction should have a token id")
		require.NotNil(t, prediction.TokenStr, "prediction should have a token")
		require.NotEmpty(t, *prediction.TokenStr, "prediction token should not be empty")
	}
}

// TestFillMask_WithTopK tests fill mask with an explicit TopK parameter.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_WithTopK(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-bert/bert-base-uncased"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	topK := 3
	resp, err := client.FillMask(
		FillMaskRequest{
			Input: "The capital of France is [MASK].",
			Parameters: &FillMaskParameters{
				TopK: &topK,
			},
		},
	)

	require.NoError(t, err, "fill mask with TopK should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have predictions")
	require.LessOrEqual(t, len(resp), topK, "should have at most TopK predictions")

	for i := range len(resp) - 1 {
		require.GreaterOrEqual(
			t,
			resp[i].Score,
			resp[i+1].Score,
			"predictions should be ordered by score (descending)",
		)
	}
}

// TestFillMask_WithTargets tests fill mask restricted to a set of target tokens.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_WithTargets(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-bert/bert-base-uncased"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	targets := []string{"paris", "london"}
	resp, err := client.FillMask(
		FillMaskRequest{
			Input: "The capital of France is [MASK].",
			Parameters: &FillMaskParameters{
				Targets: targets,
			},
		},
	)

	require.NoError(t, err, "fill mask with targets should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have predictions")

	for _, prediction := range resp {
		require.NotNil(t, prediction.TokenStr, "prediction should have a token")
		require.Contains(t, targets, *prediction.TokenStr, "prediction token should be one of the requested targets")
	}
}

// TestFillMask_BatchLiveAPI tests a batch fill mask request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-bert/bert-base-uncased"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	inputs := []string{
		"I [MASK] my dog everyday.",
		"She is a [MASK] programmer.",
		"The meeting was [MASK] due to the storm.",
	}

	resp, err := client.FillMaskBatch(
		FillMaskBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch fill mask should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, len(inputs), "response should have one entry per input")

	for i, predictions := range resp {
		require.NotEmpty(t, predictions, "each input should have predictions")

		for _, prediction := range predictions {
			require.NotEmpty(t, prediction.Sequence, "prediction should have a sequence")
			require.GreaterOrEqual(t, prediction.Score, 0.0, "score should be non-negative")
			require.LessOrEqual(t, prediction.Score, 1.0, "score should be at most 1.0")
		}

		t.Logf("Input %d: %d predictions", i, len(predictions))
	}
}

// TestFillMask_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("google-bert/bert-base-uncased"),
		WithContext(ctx),
	)

	resp, err := client.FillMask(
		FillMaskRequest{
			Input: "The capital of France is [MASK].",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestFillMask_VeryLargeBatch tests fill mask with a larger batch of inputs.
// This test requires the HF_TOKEN environment variable to be set.
func TestFillMask_VeryLargeBatch(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-bert/bert-base-uncased"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	// Create a batch of 10 inputs
	inputs := make([]string, 10)
	topics := []string{
		"I [MASK] my dog everyday.",
		"She is a [MASK] programmer.",
		"The capital of France is [MASK].",
		"The sky is [MASK] today.",
		"He [MASK] a book every night.",
		"They [MASK] to the store yesterday.",
		"The [MASK] is shining brightly.",
		"We had a [MASK] time at the party.",
		"The coffee is [MASK] and delicious.",
		"She [MASK] the guitar beautifully.",
	}
	for i, text := range topics {
		inputs[i] = text
	}

	resp, err := client.FillMaskBatch(
		FillMaskBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch fill mask with larger batch should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 10, "response should have 10 entries")

	for _, predictions := range resp {
		require.NotEmpty(t, predictions, "each input should have predictions")
	}
}
