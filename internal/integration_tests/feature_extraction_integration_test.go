//go:build integration

package integration_tests

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

// TestFeatureExtraction_LiveAPI tests a basic feature extraction request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "BAAI/bge-small-en-v1.5"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "What is the capital of France?",
		},
	)

	require.NoError(t, err, "feature extraction should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "embedding should not be empty")
}

// TestFeatureExtraction_BatchLiveAPI tests batch feature extraction against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "BAAI/bge-small-en-v1.5"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	inputs := []string{
		"The weather is nice today.",
		"I love programming in Go.",
	}

	resp, err := client.ExtractFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch feature extraction should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 2, "response should have 2 embeddings")

	for i, embedding := range resp {
		require.NotEmpty(t, embedding, "embedding %d should not be empty", i)
		t.Logf("Input %d: embedding dimension = %d", i, len(embedding))
	}
}

// TestFeatureExtraction_WithParameters tests feature extraction with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "BAAI/bge-small-en-v1.5"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	normalize := true
	truncate := true
	truncDir := hftypes.FeatureExtractionTruncationRight
	resp, err := client.ExtractFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "The capital of France is Paris, a city on the Seine.",
			Parameters: &hftypes.FeatureExtractionParameters{
				Normalize:           &normalize,
				Truncate:            &truncate,
				TruncationDirection: &truncDir,
			},
		},
	)

	require.NoError(t, err, "feature extraction with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "embedding should not be empty")
}

// TestFeatureExtraction_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("BAAI/bge-small-en-v1.5"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "What is the capital of France?",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestFeatureExtraction_BatchWithParameters tests batch feature extraction with parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_BatchWithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "BAAI/bge-small-en-v1.5"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	normalize := true
	truncate := true
	truncDir := hftypes.FeatureExtractionTruncationRight
	resp, err := client.ExtractFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{
				"The weather is nice today.",
				"I love programming in Go.",
			},
			Parameters: &hftypes.FeatureExtractionParameters{
				Normalize:           &normalize,
				Truncate:            &truncate,
				TruncationDirection: &truncDir,
			},
		},
	)

	require.NoError(t, err, "batch feature extraction with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 2, "response should have 2 embeddings")

	for i, embedding := range resp {
		require.NotEmpty(t, embedding, "embedding %d should not be empty", i)
		t.Logf("Input %d: embedding dimension = %d", i, len(embedding))
	}
}

// TestFeatureExtraction_BatchContextCancellation tests that context cancellation is respected
// for batch feature extraction.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtraction_BatchContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("BAAI/bge-small-en-v1.5"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{"The weather is nice today.", "I love programming in Go."},
		},
	)

	require.Error(t, err, "batch request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
