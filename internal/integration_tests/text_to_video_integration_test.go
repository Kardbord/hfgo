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
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

const textToVideoModel = "ZhengmingYu/DMAD"

// TestGenerateVideo_LiveAPI tests a basic text-to-video request against the
// live HF API using the WaveSpeed provider.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateVideo_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(textToVideoModel),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newWavespeedProvider(t)),
		hfopts.WithMaxResponseBodyBytes(16<<20), // Videos can be large.
	)

	response, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: "A serene mountain landscape at sunset",
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

// TestGenerateVideo_ContextCancellation tests that context cancellation is
// respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestGenerateVideo_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(textToVideoModel),
		hfopts.WithContext(ctx),
		hfopts.WithProvider(newWavespeedProvider(t)),
	)

	response, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: "A serene mountain landscape at sunset",
	})

	require.Error(t, err, "request with cancelled context should fail")
	require.Equal(
		t,
		hftypes.TextToVideoResponse{},
		response,
		"response should be zero for cancelled context",
	)
}
