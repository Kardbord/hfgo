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

const detectObjectsImageURL = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"

// detectObjectsInput fetches the test image and returns it as base64-encoded
// data, as required by the object detection task.
func detectObjectsInput(t *testing.T) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, detectObjectsImageURL, http.NoBody)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	return base64.StdEncoding.EncodeToString(body)
}

// TestDetectObjects_LiveAPI tests a basic object detection request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestDetectObjects_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/detr-resnet-50"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
		Input: detectObjectsInput(t),
	})

	require.NoError(t, err, "object detection should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have detections")

	for _, det := range resp {
		require.NotEmpty(t, det.Label, "detection should have a label")
		require.GreaterOrEqual(t, det.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, det.Score, 1.0, "score should be at most 1.0")
		require.GreaterOrEqual(t, det.Box.XMin, 0, "box coordinates should be non-negative")
		require.GreaterOrEqual(t, det.Box.YMin, 0, "box coordinates should be non-negative")
		require.LessOrEqual(t, det.Box.XMin, det.Box.XMax, "xmin should not exceed xmax")
		require.LessOrEqual(t, det.Box.YMin, det.Box.YMax, "ymin should not exceed ymax")
	}
}

// TestDetectObjects_WithThreshold tests object detection with an explicit
// threshold parameter.
// This test requires the HF_TOKEN environment variable to be set.
func TestDetectObjects_WithThreshold(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/detr-resnet-50"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	threshold := 0.8
	resp, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
		Input: detectObjectsInput(t),
		Parameters: &hftypes.ObjectDetectionParameters{
			Threshold: &threshold,
		},
	})

	require.NoError(t, err, "object detection with threshold should succeed")
	require.NotNil(t, resp, "response should not be nil")

	// Every returned detection should meet the requested threshold.
	for _, det := range resp {
		require.GreaterOrEqual(t, det.Score, threshold, "score should meet threshold")
	}
}

// TestDetectObjects_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestDetectObjects_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("facebook/detr-resnet-50"),
		hfopts.WithContext(ctx),
	)

	// Use a minimal inline base64 image (1x1 transparent PNG) so the cancelled
	// context is exercised immediately, without a preliminary HTTP fetch.
	resp, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
		Input: testutils.TinyPNGBase64,
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
