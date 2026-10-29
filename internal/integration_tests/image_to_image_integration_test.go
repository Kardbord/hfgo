//go:build integration

package integration_tests

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const imageToImageModel = "black-forest-labs/FLUX.2-klein-base-4B"

// TestGenerateImageToImage_LiveAPI tests a basic image-to-image request against
// the live HF API using the WaveSpeed provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImageToImage_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(imageToImageModel),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newWavespeedProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20), // Images can be large.
	)

	response, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: testutils.TinyPNGBase64,
		Parameters: &hftypes.ImageToImageParameters{
			Prompt: new("Make the image blue"),
		},
	})

	require.NoError(t, err, "image-to-image should succeed")
	require.NotEmpty(t, response.Image, "generated image should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "image/"),
		"expected an image media type, got %s",
		response.MediaType,
	)

	imageType := http.DetectContentType(response.Image)
	require.True(
		t,
		strings.HasPrefix(imageType, "image/"),
		"expected image data, got %s",
		imageType,
	)
}

// TestGenerateImageToImage_ContextCancellation tests that context cancellation
// is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImageToImage_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(imageToImageModel),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newWavespeedProvider(t)),
	)

	response, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: testutils.TinyPNGBase64,
		Parameters: &hftypes.ImageToImageParameters{
			Prompt: new("Make the image blue"),
		},
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Equal(
		t,
		hftypes.ImageToImageResponse{},
		response,
		"response should be zero for cancelled context",
	)
}
