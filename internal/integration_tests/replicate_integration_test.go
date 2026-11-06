//go:build integration

package integration_tests

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// newReplicateProvider returns a [hfproviders.ReplicateProvider] backed by a
// Hub model resolver for live API tests. Callers gate the test on HF_TOKEN.
func newReplicateProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
	)

	provider, err := hfproviders.NewReplicateProvider(resolver)
	require.NoError(t, err)

	return provider
}

const (
	replicateImageToImageModel = "black-forest-labs/FLUX.2-klein-base-4B"
	replicateTestImageURL      = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"
)

// TestGenerateImage_ReplicateLiveAPI tests a basic text-to-image request
// against the live HF API using the Replicate provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_ReplicateLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("black-forest-labs/FLUX.1-dev"),
		hfopts.WithProvider(newReplicateProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	response := testutils.RetryTransient(ctx, t, func() (hftypes.TextToImageResponse, error) {
		return client.GenerateImage(ctx, hftypes.TextToImageRequest{
			Input: "A serene mountain landscape at sunset",
			Parameters: &hftypes.TextToImageParameters{
				Width:  new(512),
				Height: new(512),
			},
		})
	})

	require.NotEmpty(t, response.Image, "generated image should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "image/"),
		"expected an image media type, got %s",
		response.MediaType,
	)
}

// TestGenerateImageToImage_ReplicateLiveAPI tests a basic image-to-image
// request against the live HF API using the Replicate provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImageToImage_ReplicateLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(replicateImageToImageModel),
		hfopts.WithProvider(newReplicateProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "Remove the glasses"

	response := testutils.RetryTransient(ctx, t, func() (hftypes.ImageToImageResponse, error) {
		return client.GenerateImageToImage(ctx, hftypes.ImageToImageRequest{
			Input: replicateTestImageURL,
			Parameters: &hftypes.ImageToImageParameters{
				Prompt: new(prompt),
			},
		})
	})

	require.NotEmpty(t, response.Image, "generated image should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "image/"),
		"expected an image media type, got %s",
		response.MediaType,
	)
}

// TestGenerateVideo_ReplicateLiveAPI tests a basic text-to-video request
// against the live HF API using the Replicate provider. Video generation can
// take several minutes, so the context deadline is generous.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateVideo_ReplicateLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
		hfopts.WithProvider(newReplicateProvider(t)),
		hfopts.WithMaxResponseBodyBytes(64<<20),
	)

	response := testutils.RetryTransient(ctx, t, func() (hftypes.TextToVideoResponse, error) {
		return client.GenerateVideo(ctx, hftypes.TextToVideoRequest{
			Input: "A young man walking on the street",
		})
	})

	require.NotEmpty(t, response.Video, "generated video should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "video/"),
		"expected a video media type, got %s",
		response.MediaType,
	)
}

// TestRecognizeSpeech_ReplicateLiveAPI tests a basic speech recognition
// request against the live HF API using the Replicate provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestRecognizeSpeech_ReplicateLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithProvider(newReplicateProvider(t)),
	)

	// Replicate passes HTTP(S) audio URLs through unchanged, so the sample
	// needs no base64 round trip.
	resp := testutils.RetryTransient(ctx, t, func() (hftypes.SpeechRecognitionResponse, error) {
		return client.RecognizeSpeech(ctx, hftypes.SpeechRecognitionRequest{
			Input: audioURL,
		})
	})

	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
}
