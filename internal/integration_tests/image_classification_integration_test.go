//go:build integration

package integration_tests

import (
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const classifyImageURL = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"

// classifyImageInput fetches the test image and returns it as base64-encoded
// data, as required by the image classification task.
func classifyImageInput(t *testing.T) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, classifyImageURL, http.NoBody)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	return base64.StdEncoding.EncodeToString(body)
}

// TestClassifyImage_LiveAPI tests a basic image classification request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestClassifyImage_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google/vit-base-patch16-224"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: classifyImageInput(t),
	})

	require.NoError(t, err, "image classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have predictions")

	for _, prediction := range resp {
		require.NotEmpty(t, prediction.Label, "prediction should have a label")
		require.GreaterOrEqual(t, prediction.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, prediction.Score, 1.0, "score should be at most 1.0")
	}
}

// TestClassifyImage_WithParameters tests image classification with explicit
// function_to_apply and top_k parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestClassifyImage_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google/vit-base-patch16-224"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: classifyImageInput(t),
		Parameters: &hftypes.ImageClassificationParameters{
			FunctionToApply: new(hftypes.ImageClassificationFuncSoftmax),
			TopK:            new(5),
		},
	})

	require.NoError(t, err, "image classification with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.LessOrEqual(t, len(resp), 5, "top_k should limit the number of predictions")
}

// TestClassifyImage_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestClassifyImage_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("google/vit-base-patch16-224"),
		hfopts.WithContext(ctx),
	)

	// Use a minimal inline base64 image (1x1 transparent PNG) so the cancelled
	// context is exercised immediately, without a preliminary HTTP fetch.
	resp, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: testutils.TinyPNGBase64,
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
