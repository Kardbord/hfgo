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
	"github.com/stretchr/testify/require"
)

const audioURL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

// speechRecognitionInput fetches the test audio and returns it as base64-encoded
// data, as required by the speech recognition task.
func speechRecognitionInput(t *testing.T) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, audioURL, http.NoBody)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	return base64.StdEncoding.EncodeToString(body)
}

// TestRecognizeSpeech_LiveAPI tests a basic speech recognition request against
// the live HF API. This test requires the HF_TOKEN environment variable to be
// set.
func TestRecognizeSpeech_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "openai/whisper-large-v3"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: speechRecognitionInput(t),
	})

	require.NoError(t, err, "speech recognition should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
}

// TestRecognizeSpeech_WithParameters tests speech recognition with the
// return_timestamps parameter.
// This test requires the HF_TOKEN environment variable to be set.
func TestRecognizeSpeech_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "openai/whisper-large-v3"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: speechRecognitionInput(t),
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	})

	require.NoError(t, err, "speech recognition with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp.Text, "recognized text should not be empty")
}

// TestRecognizeSpeech_ContextCancellation tests that context cancellation is
// respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestRecognizeSpeech_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: "AAAA",
	})

	require.Error(t, err, "request with cancelled context should cancel")
	require.Empty(t, resp)
}
