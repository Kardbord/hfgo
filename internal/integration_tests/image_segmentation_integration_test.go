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

const segmentImageImageURL = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"

// segmentImageInput fetches the test image and returns it as base64-encoded
// data, as required by the image segmentation task.
func segmentImageInput(t *testing.T) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, segmentImageImageURL, http.NoBody)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	return base64.StdEncoding.EncodeToString(body)
}

// TestSegmentImage_LiveAPI tests a basic image segmentation request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestSegmentImage_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "facebook/mask2former-swin-large-coco-panoptic"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: segmentImageInput(t),
	})

	require.NoError(t, err, "image segmentation should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have segments")

	for _, segment := range resp {
		require.NotEmpty(t, segment.Label, "segment should have a label")
		require.NotEmpty(t, segment.Mask, "segment should have a mask")
	}
}

// TestSegmentImage_WithParameters tests image segmentation with explicit
// mask and overlap thresholds.
// This test requires the HF_TOKEN environment variable to be set.
func TestSegmentImage_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "mattmdjaga/segformer_b2_clothes"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: segmentImageInput(t),
		Parameters: &hftypes.ImageSegmentationParameters{
			MaskThreshold:            new(0.5),
			OverlapMaskAreaThreshold: new(0.5),
			Threshold:                new(0.5),
		},
	})

	require.NoError(t, err, "image segmentation with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have segments")
}

// TestSegmentImage_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestSegmentImage_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("mattmdjaga/segformer_b2_clothes"),
		hfopts.WithContext(ctx),
	)

	// Use a minimal inline base64 image (1x1 transparent PNG) so the cancelled
	// context is exercised immediately, without a preliminary HTTP fetch.
	resp, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: testutils.TinyPNGBase64,
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
