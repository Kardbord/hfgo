//go:build integration

package hfgo

import (
	"bytes"
	"os"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestSpeechRecognition_Recognize(t *testing.T) {
	t.Parallel()

	token := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, token, "HF_TOKEN must be set for integration tests")

	client := NewClient(
		WithToken(token),
		WithModel("openai/whisper-large-v3"),
	)

	result, err := client.RecognizeSpeech(SpeechRecognitionRequest{
		Input: "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
	})
	require.NoError(t, err)
	require.NotEmpty(t, result.Text)
}

func TestSpeechRecognition_RecognizeReader(t *testing.T) {
	t.Parallel()

	token := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, token, "HF_TOKEN must be set for integration tests")

	client := NewClient(
		WithToken(token),
		WithModel("openai/whisper-large-v3"),
	)

	reader := bytes.NewReader(testutils.DecodeTestAudioWAV())
	result, err := client.RecognizeSpeechReader(reader, nil)
	require.NoError(t, err)
	require.NotEmpty(t, result.Text)
}

func TestSpeechRecognition_RecognizeReaderWithParams(t *testing.T) {
	t.Parallel()

	token := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, token, "HF_TOKEN must be set for integration tests")

	client := NewClient(
		WithToken(token),
		WithModel("openai/whisper-large-v3"),
	)

	reader := bytes.NewReader(testutils.DecodeTestAudioWAV())
	result, err := client.RecognizeSpeechReader(reader, &SpeechRecognitionRequestParameters{
		ReturnTimestamps: Ptr(true),
	})
	require.NoError(t, err)
	require.NotEmpty(t, result.Text)
}

func TestSpeechRecognition_RecognizeBatch(t *testing.T) {
	t.Parallel()

	token := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, token, "HF_TOKEN must be set for integration tests")

	client := NewClient(
		WithToken(token),
		WithModel("openai/whisper-large-v3"),
	)

	result, err := client.RecognizeSpeechBatch(SpeechRecognitionBatchRequest{
		Inputs: []string{
			testutils.TestAudioWAV,
			testutils.TestAudioWAV,
		},
	})
	require.NoError(t, err)
	require.Len(t, result, 1)
	require.NotEmpty(t, result[0].Text)
}
