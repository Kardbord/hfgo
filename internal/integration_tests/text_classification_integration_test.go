//go:build integration

package hfgo

import (
	"context"
	"os"
	"testing"
	"time"

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

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	const text = "This product is excellent and I love it!"
	resp, err := client.ClassifyText(
		TextClassificationRequest{
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

// TestTextClassification_BatchLiveAPI tests batch text classification against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "ProsusAI/finbert"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	inputs := []string{
		"This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end.",
		"This could have been better. The director was completely unfaithful to the books, and it moved at a snail's pace.",
		"I enjoyed this film, though it had some minor flaws.",
	}

	resp, err := client.ClassifyTextBatch(
		TextClassificationBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch text classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 3, "response should have 3 classifications")

	for i, classifications := range resp {
		require.NotEmpty(t, classifications, "each input should have classifications")

		for _, classification := range classifications {
			require.NotEmpty(t, classification.Label, "each classification should have a label")
			require.GreaterOrEqual(t, classification.Score, 0.0, "score should be non-negative")
			require.LessOrEqual(t, classification.Score, 1.0, "score should be at most 1.0")
		}

		t.Logf("Input %d: %d classifications", i, len(classifications))
	}
}

// TestTextClassification_WithParameters tests text classification with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "ProsusAI/finbert"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	topK := 2
	function := TextClassificationFuncSoftmax
	resp, err := client.ClassifyText(
		TextClassificationRequest{
			Input: "Excellent product!",
			Parameters: &TextClassificationParameters{
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

	client := NewClient(
		WithToken(apiToken),
		WithModel("ProsusAI/finbert"),
		WithContext(ctx),
	)

	resp, err := client.ClassifyText(
		TextClassificationRequest{
			Input: "This is great!",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestTextClassification_VeryLargeBatch tests classification with a larger batch of inputs.
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_VeryLargeBatch(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("ProsusAI/finbert"),
		WithContext(ctx),
	)

	// Create a batch of 10 inputs
	inputs := make([]string, 10)
	sentiments := []string{
		"excellent",
		"terrible",
		"amazing",
		"awful",
		"wonderful",
		"bad",
		"fantastic",
		"poor",
		"great",
		"disappointing",
	}
	for i, sentiment := range sentiments {
		inputs[i] = "This product is " + sentiment + "!"
	}

	resp, err := client.ClassifyTextBatch(
		TextClassificationBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch text classification with larger batch should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 10, "response should have 10 classifications")

	for _, classifications := range resp {
		require.NotEmpty(t, classifications, "each input should have classifications")
	}
}

// TestTextClassification_TopKResponseFormatQuirk documents the HuggingFace API quirk where
// the response format differs based on whether the TopK parameter is explicitly set.
//
// This test documents an important API quirk:
//   - When TopK is explicitly set (e.g., to 1, 2, or any value), the API returns
//     the expected per-input format: [[input1_classifications], [input2_classifications], ...]
//   - When TopK is unset (nil), the API returns a flat format: [[all_classifications_together]]
//
// The SDK handles this transparently: normalizeTextClassificationResponse() is only called
// when TopK is unset (nil), ensuring the flat format is reshaped to per-input format.
// This conditional normalization prevents data corruption when TopK is explicitly set.
//
// This test requires the HF_TOKEN environment variable to be set.
func TestTextClassification_TopKResponseFormatQuirk(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "ProsusAI/finbert"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	inputs := []string{
		"This product is amazing!",
		"This product is terrible!",
		"This product is okay.",
	}

	// Test 1: Without TopK (unset) - API returns flat format which the SDK normalizes to per-input format
	t.Run("without_topk_triggers_normalization", func(t *testing.T) {
		resp, err := client.ClassifyTextBatch(
			TextClassificationBatchRequest{
				Inputs: inputs,
				// TopK is not set (nil)
			},
		)

		require.NoError(t, err, "batch text classification without TopK should succeed")
		require.NotNil(t, resp, "response should not be nil")
		// The SDK normalizes the flat format to per-input format
		require.Len(t, resp, 3, "response should have 3 elements (one per input)")

		for i, classifications := range resp {
			require.NotEmpty(t, classifications, "each input should have at least one classification")
			// Without TopK, we expect 1 classification per input (the top one)
			require.Len(t, classifications, 1, "without TopK, each input should have 1 classification")

			classification := classifications[0]
			require.NotEmpty(t, classification.Label, "classification should have a label")
			require.GreaterOrEqual(t, classification.Score, 0.0, "score should be non-negative")
			require.LessOrEqual(t, classification.Score, 1.0, "score should be at most 1.0")

			t.Logf("Input %d: %s (score: %.4f)", i, classification.Label, classification.Score)
		}
	})

	// Test 2: With TopK=1 - API returns per-input format; SDK skips normalization
	t.Run("with_topk_1_skips_normalization", func(t *testing.T) {
		topK := 1
		resp, err := client.ClassifyTextBatch(
			TextClassificationBatchRequest{
				Inputs: inputs,
				Parameters: &TextClassificationParameters{
					TopK: &topK,
				},
			},
		)

		require.NoError(t, err, "batch text classification with TopK=1 should succeed")
		require.NotNil(t, resp, "response should not be nil")
		require.Len(t, resp, 3, "response should have 3 elements (one per input)")

		for i, classifications := range resp {
			require.NotEmpty(t, classifications, "each input should have classifications")
			require.Len(t, classifications, 1, "with TopK=1, each input should have 1 classification")

			classification := classifications[0]
			require.NotEmpty(t, classification.Label, "classification should have a label")

			t.Logf("Input %d: %s (score: %.4f)", i, classification.Label, classification.Score)
		}
	})

	// Test 3: With TopK=2 - API returns per-input format with multiple classifications; SDK skips normalization
	t.Run("with_topk_2_skips_normalization", func(t *testing.T) {
		topK := 2
		resp, err := client.ClassifyTextBatch(
			TextClassificationBatchRequest{
				Inputs: inputs,
				Parameters: &TextClassificationParameters{
					TopK: &topK,
				},
			},
		)

		require.NoError(t, err, "batch text classification with TopK=2 should succeed")
		require.NotNil(t, resp, "response should not be nil")
		require.Len(t, resp, 3, "response should have 3 elements (one per input)")

		for i, classifications := range resp {
			require.NotEmpty(t, classifications, "each input should have classifications")
			require.LessOrEqual(t, len(classifications), topK, "each input should have at most TopK classifications")

			for j, classification := range classifications {
				require.NotEmpty(t, classification.Label, "classification should have a label")
				require.GreaterOrEqual(t, classification.Score, 0.0, "score should be non-negative")
				require.LessOrEqual(t, classification.Score, 1.0, "score should be at most 1.0")

				t.Logf("Input %d, Classification %d: %s (score: %.4f)", i, j, classification.Label, classification.Score)
			}

			// Verify classifications are ordered by score (descending)
			for j := range len(classifications) - 1 {
				require.GreaterOrEqual(
					t,
					classifications[j].Score,
					classifications[j+1].Score,
					"classifications should be ordered by score (descending)",
				)
			}
		}
	})
}
