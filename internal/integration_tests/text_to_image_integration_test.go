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
	"github.com/stretchr/testify/require"
)

const textToImageModel = "stabilityai/stable-diffusion-3-medium-diffusers"

// TestGenerateImage_LiveAPI tests a basic text-to-image request against the
// live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(textToImageModel),
		hfopts.WithContext(ctx),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	response, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: "A serene mountain landscape at sunset",
		Parameters: &hftypes.TextToImageParameters{
			Width:  new(512),
			Height: new(512),
		},
	})

	require.NoError(t, err, "text-to-image should succeed")
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

// TestGenerateImage_ContextCancellation tests that context cancellation is
// respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(textToImageModel),
		hfopts.WithContext(ctx),
	)

	response, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: "A serene mountain landscape at sunset",
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Equal(
		t,
		hftypes.TextToImageResponse{},
		response,
		"response should be zero for cancelled context",
	)
}
