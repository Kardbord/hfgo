//go:build integration

package hfgo

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestZeroShotTextClassification_LiveAPI tests a basic zero-shot text classification
// against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-mnli"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	const text = "This product is excellent and I love it!"
	candidateLabels := []string{"positive", "negative", "neutral"}

	resp, err := client.ZeroShotClassifyText(
		ZeroShotTextClassificationRequest{
			Input: text,
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")
}

// TestZeroShotTextClassification_BatchLiveAPI tests batch zero-shot text classification
// against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-mnli"

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

	candidateLabels := []string{"positive", "negative", "neutral"}

	resp, err := client.ZeroShotClassifyTextBatch(
		ZeroShotTextClassificationBatchRequest{
			Inputs: inputs,
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "batch zero-shot text classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 3, "response should have 3 classifications")

	for i, classifications := range resp {
		require.Len(
			t,
			classifications,
			len(candidateLabels),
			"each input should have classifications",
		)
		t.Logf("Input %d: %d classifications", i, len(classifications))
	}
}

// TestZeroShotTextClassification_WithHypothesisTemplate tests zero-shot text classification
// with a custom hypothesis template.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_WithHypothesisTemplate(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-mnli"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	candidateLabels := []string{"positive", "negative"}
	hypothesisTemplate := "This example is {}."

	resp, err := client.ZeroShotClassifyText(
		ZeroShotTextClassificationRequest{
			Input: "I love this product!",
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels:    candidateLabels,
				HypothesisTemplate: &hypothesisTemplate,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification with hypothesis template should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")
}

// TestZeroShotTextClassification_WithMultiLabel tests zero-shot text classification
// with multi-label mode enabled.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_WithMultiLabel(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-mnli"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	candidateLabels := []string{"positive", "negative", "neutral"}
	multiLabel := true

	resp, err := client.ZeroShotClassifyText(
		ZeroShotTextClassificationRequest{
			Input: "This product is great and I love it!",
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
				MultiLabel:      &multiLabel,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification with multi-label should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")
}

// TestZeroShotTextClassification_VeryLargeBatch tests classification with a larger batch of inputs.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_VeryLargeBatch(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("facebook/bart-large-mnli"),
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

	candidateLabels := []string{"positive", "negative", "neutral"}

	resp, err := client.ZeroShotClassifyTextBatch(
		ZeroShotTextClassificationBatchRequest{
			Inputs: inputs,
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "batch zero-shot text classification with larger batch should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, 10, "response should have 10 classifications")

	for _, classifications := range resp {
		require.Len(t, classifications, len(candidateLabels))
	}
}

// TestZeroShotTextClassification_MultipleCandidateLabels tests zero-shot classification
// with a larger set of candidate labels.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_MultipleCandidateLabels(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("facebook/bart-large-mnli"),
		WithContext(ctx),
	)

	// More comprehensive set of candidate labels
	candidateLabels := []string{
		"positive",
		"negative",
		"neutral",
		"excited",
		"disappointed",
		"confused",
	}

	resp, err := client.ZeroShotClassifyText(
		ZeroShotTextClassificationRequest{
			Input: "I absolutely love this product! It exceeded all my expectations!",
			Parameters: &ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification with multiple labels should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, len(candidateLabels), "response should have classifications")
}
