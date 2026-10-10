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

// TestSummarization_LiveAPI tests a basic summarization request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestSummarization_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-cnn"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	const input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. " +
		"Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel " +
		"Tower surpassed the Washington Monument to become the tallest man-made structure in the world."

	resp, err := client.Summarize(
		hftypes.SummarizationRequest{
			Input: input,
		},
	)

	require.NoError(t, err, "summarization should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have content")
	require.NotEmpty(t, resp[0].SummaryText, "summary text should not be empty")
}

// TestSummarization_WithParameters tests summarization with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestSummarization_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/bart-large-cnn"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	cleanUp := true
	truncation := hftypes.SummarizationTruncationOnlyFirst
	input := "The industrial revolution transformed agriculture, manufacturing, mining, and transport, " +
		"leading to massive social and economic changes across the world."
	resp, err := client.Summarize(
		hftypes.SummarizationRequest{
			Input: input,
			Parameters: &hftypes.SummarizationParameters{
				CleanUpTokenizationSpaces: &cleanUp,
				Truncation:                &truncation,
			},
		},
	)

	require.NoError(t, err, "summarization with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have content")
	require.NotEmpty(t, resp[0].SummaryText, "summary text should not be empty")
}

// TestSummarization_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestSummarization_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("facebook/bart-large-cnn"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.Summarize(
		hftypes.SummarizationRequest{
			Input: "Some text that should be summarized.",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
