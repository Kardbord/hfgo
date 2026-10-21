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

// TestFeatureExtractionPooled_LiveAPI tests a basic pooled feature extraction
// request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooled_LiveAPI(t *testing.T) {
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

	resp, err := client.ExtractPooledFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "What is the capital of France?",
		},
	)

	require.NoError(t, err, "feature extraction should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "embedding should not be empty")
}

// TestFeatureExtractionPooledBatch_LiveAPI tests batch pooled feature
// extraction against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooledBatch_LiveAPI(t *testing.T) {
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

	resp, err := client.ExtractPooledFeaturesBatch(
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

// TestFeatureExtractionPooled_WithParameters tests pooled feature extraction
// with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooled_WithParameters(t *testing.T) {
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
	resp, err := client.ExtractPooledFeatures(
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

// TestFeatureExtractionPooled_ContextCancellation tests that context
// cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooled_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("BAAI/bge-small-en-v1.5"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractPooledFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "What is the capital of France?",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestFeatureExtractionPooledBatch_WithParameters tests batch pooled feature
// extraction with parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooledBatch_WithParameters(t *testing.T) {
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
	resp, err := client.ExtractPooledFeaturesBatch(
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

// TestFeatureExtractionPooledBatch_ContextCancellation tests that context
// cancellation is respected for batch pooled feature extraction.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionPooledBatch_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("BAAI/bge-small-en-v1.5"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractPooledFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{"The weather is nice today.", "I love programming in Go."},
		},
	)

	require.Error(t, err, "batch request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestFeatureExtractionToken_LiveAPI tests token-level feature extraction
// against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionToken_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-base"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractTokenFeatures(
		hftypes.FeatureExtractionRequest{
			Input: "Transformers is an awesome library!",
		},
	)
	require.NoError(t, err, "token-level feature extraction should succeed")
	require.NotEmpty(t, resp, "token embeddings should not be empty")
	require.Greater(t, len(resp), 1, "input should produce multiple tokens")

	hiddenSize := len(resp[0])
	require.Positive(t, hiddenSize, "token embedding should not be empty")
	for i, token := range resp {
		require.Len(t, token, hiddenSize, "token %d should have the model hidden size", i)
	}
}

// TestFeatureExtractionToken_BatchLiveAPI tests batched token-level feature
// extraction against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestFeatureExtractionToken_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-base"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ExtractTokenFeaturesBatch(
		hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{
				"Transformers is an awesome library!",
				"Token-level embeddings are returned for every token.",
			},
		},
	)
	require.NoError(t, err, "batch token-level feature extraction should succeed")
	require.Len(t, resp, 2, "response should contain token embeddings for each input")

	var hiddenSize int
	for i, input := range resp {
		require.Greater(t, len(input), 1, "input %d should produce multiple tokens", i)
		for j, token := range input {
			require.NotEmpty(t, token, "input %d token %d should not be empty", i, j)
			if hiddenSize == 0 {
				hiddenSize = len(token)
			}
			require.Len(t, token, hiddenSize, "all token vectors should share the hidden size")
		}
	}
}
