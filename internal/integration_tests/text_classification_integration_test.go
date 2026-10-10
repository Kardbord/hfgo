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

// TestTextClassification_LiveAPI tests a basic text classification against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "ProsusAI/finbert"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	const text = "This product is excellent and I love it!"
	resp, err := client.ClassifyText(
		hftypes.TextClassificationRequest{
			Input: text,
		},
	)

	require.NoError(t, err, "text classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")

	classification := resp[0]
	require.NotEmpty(t, classification.Label, "classification should have a label")
	require.GreaterOrEqual(t, classification.Score, 0.0, "score should be non-negative")
	require.LessOrEqual(t, classification.Score, 1.0, "score should be at most 1.0")
}

// TestTextClassification_WithParameters tests text classification with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "ProsusAI/finbert"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	topK := 2
	function := hftypes.TextClassificationFuncSoftmax
	resp, err := client.ClassifyText(
		hftypes.TextClassificationRequest{
			Input: "Excellent product!",
			Parameters: &hftypes.TextClassificationParameters{
				TopK:            &topK,
				FunctionToApply: &function,
			},
		},
	)

	require.NoError(t, err, "text classification with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")

	// Verify we got the requested number of top classifications
	require.LessOrEqual(t, len(resp), topK, "should have at most TopK classifications")

	// Verify classifications are ordered by score (descending)
	for i := range len(resp) - 1 {
		require.GreaterOrEqual(
			t,
			resp[i].Score,
			resp[i+1].Score,
			"classifications should be ordered by score",
		)
	}
}

// TestTextClassification_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("ProsusAI/finbert"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ClassifyText(
		hftypes.TextClassificationRequest{
			Input: "This is great!",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
