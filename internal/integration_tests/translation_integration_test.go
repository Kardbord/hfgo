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

// TestTranslation_LiveAPI tests a basic translation request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTranslation_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-t5/t5-small"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.Translate(
		hftypes.TranslationRequest{
			Input: "Hello, how are you doing today?",
		},
	)

	require.NoError(t, err, "translation should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have content")
	require.NotEmpty(t, resp[0].TranslationText, "translation text should not be empty")
}

// TestTranslation_WithParameters tests translation with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestTranslation_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google-t5/t5-small"

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	cleanUp := true
	truncation := hftypes.TranslationTruncationOnlyFirst
	resp, err := client.Translate(
		hftypes.TranslationRequest{
			Input: "The weather is lovely and sunny today.",
			Parameters: &hftypes.TranslationParameters{
				CleanUpTokenizationSpaces: &cleanUp,
				Truncation:                &truncation,
			},
		},
	)

	require.NoError(t, err, "translation with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have content")
	require.NotEmpty(t, resp[0].TranslationText, "translation text should not be empty")
}

// TestTranslation_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestTranslation_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("google-t5/t5-small"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.Translate(
		hftypes.TranslationRequest{
			Input: "Hello, how are you?",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
