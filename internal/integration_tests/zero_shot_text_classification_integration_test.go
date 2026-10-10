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

// TestZeroShotTextClassification_LiveAPI tests a basic zero-shot text classification
// against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-mnli"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	const text = "This product is excellent and I love it!"
	candidateLabels := []string{"positive", "negative", "neutral"}

	resp, err := client.ZeroShotClassifyText(
		hftypes.ZeroShotTextClassificationRequest{
			Input: text,
			Parameters: &hftypes.ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")
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

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	candidateLabels := []string{"positive", "negative"}
	hypothesisTemplate := "This example is {}."

	resp, err := client.ZeroShotClassifyText(
		hftypes.ZeroShotTextClassificationRequest{
			Input: "I love this product!",
			Parameters: &hftypes.ZeroShotTextClassificationParameters{
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

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	candidateLabels := []string{"positive", "negative", "neutral"}
	multiLabel := true

	resp, err := client.ZeroShotClassifyText(
		hftypes.ZeroShotTextClassificationRequest{
			Input: "This product is great and I love it!",
			Parameters: &hftypes.ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
				MultiLabel:      &multiLabel,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification with multi-label should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have classifications")
}

// TestZeroShotTextClassification_MultipleCandidateLabels tests zero-shot classification
// with a larger set of candidate labels.
// This test requires the HF_TOKEN environment variable to be set.
func TestZeroShotTextClassification_MultipleCandidateLabels(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("facebook/bart-large-mnli"),
		hfopts.WithContext(ctx),
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
		hftypes.ZeroShotTextClassificationRequest{
			Input: "I absolutely love this product! It exceeded all my expectations!",
			Parameters: &hftypes.ZeroShotTextClassificationParameters{
				CandidateLabels: candidateLabels,
			},
		},
	)

	require.NoError(t, err, "zero-shot text classification with multiple labels should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, len(candidateLabels), "response should have classifications")
}
