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
	"github.com/stretchr/testify/require"
)

// newFalAIProvider returns a [hfproviders.FalAIProvider] backed by a Hub
// model resolver for live API tests. Callers gate the test on HF_TOKEN.
func newFalAIProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
	)

	provider, err := hfproviders.NewFalAIProvider(resolver)
	require.NoError(t, err)

	return provider
}

const (
	falaiImageToImageModel = "black-forest-labs/FLUX.2-klein-9B"
	falaiTestImageURL      = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"
)

// TestGenerateImage_FalAILiveAPI tests a basic text-to-image request against
// the live HF API using the Fal AI provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_FalAILiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("krea/Krea-2-Turbo"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
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
	require.NotEmpty(t, response.ProviderMetadata["request_id"], "a request id should surface")
}

// TestGenerateImageToImage_FalAILiveAPI tests a basic image-to-image request
// against the live HF API using the Fal AI provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImageToImage_FalAILiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(falaiImageToImageModel),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "Remove the glasses"

	response, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: falaiTestImageURL,
		Parameters: &hftypes.ImageToImageParameters{
			Prompt: new(prompt),
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
}

// TestGenerateVideo_FalAILiveAPI tests a basic text-to-video request against
// the live HF API using the Fal AI provider. Video generation can take
// several minutes, so the context deadline is generous.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateVideo_FalAILiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("Wan-AI/Wan2.2-TI2V-5B"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
		hfopts.WithMaxResponseBodyBytes(64<<20),
	)

	response, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: "A young man walking on the street",
	})

	require.NoError(t, err, "text-to-video should succeed")
	require.NotEmpty(t, response.Video, "generated video should not be empty")
	require.True(
		t,
		strings.HasPrefix(response.MediaType, "video/"),
		"expected a video media type, got %s",
		response.MediaType,
	)
}

// TestSegmentImage_FalAILiveAPI tests a basic image-segmentation request
// against the live HF API using the Fal AI provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestSegmentImage_FalAILiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("briaai/RMBG-2.0"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	segments, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: falaiTestImageURL,
	})

	require.NoError(t, err, "image segmentation should succeed")
	require.NotEmpty(t, segments, "expected at least one segment")
	for _, segment := range segments {
		require.Equal(t, "mask", segment.Label, "fal reports an anonymous mask")
		require.NotEmpty(t, segment.Mask, "mask should not be empty")
	}
}

// TestRecognizeSpeech_FalAILiveAPI tests a basic speech recognition request
// against the live HF API using the Fal AI provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestRecognizeSpeech_FalAILiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
	)

	// fal passes HTTP(S) audio URLs through unchanged, so the sample needs
	// no base64 round trip.
	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioURL,
	})

	require.NoError(t, err, "speech recognition should succeed")
	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
}

// TestRecognizeSpeech_FalAILiveAPIWithTimestamps tests speech recognition
// with the return_timestamps parameter against the live HF API using the
// Fal AI provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestRecognizeSpeech_FalAILiveAPIWithTimestamps(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
	)

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioURL,
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	})

	require.NoError(t, err, "speech recognition with timestamps should succeed")
	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
	require.NotEmpty(t, resp.Chunks, "chunk timestamps should be returned")
}

// TestGenerateImage_FalAIContextCancellation tests that context cancellation
// is respected by the Fal AI queue exchange.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateImage_FalAIContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("krea/Krea-2-Turbo"),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newFalAIProvider(t)),
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
