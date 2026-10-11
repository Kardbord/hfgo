// Example: Automatic Speech Recognition (Speech Recognition)
package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const audioURL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("openai/whisper-large-v3"),
	)

	fmt.Println("Fetching sample audio: " + audioURL)
	fmt.Println("...")

	audioInput, err := fetchAudio()
	if err != nil {
		log.Fatalf("Failed to fetch audio: %v\n", err)
	}

	// The codec auto-detects Content-Type from decoded bytes. The example audio
	// (jfk.flac) is sniffed as application/octet-stream, so we set the correct
	// type explicitly. Caller-supplied headers always take precedence.
	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioInput,
	}, hfopts.WithHeader("Content-Type", "audio/flac"))
	if err != nil {
		log.Fatalf("error running speech recognition: %v\n", err)
	}

	fmt.Println("Recognized text:")
	fmt.Println(resp.Text)
}

// mustRequest creates an HTTP GET request with context. It panics if request creation fails.
func mustRequest(url string) *http.Request {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, http.NoBody)
	if err != nil {
		panic(err)
	}

	return req
}

// fetchAudio retrieves the audio and returns it as base64-encoded data.
func fetchAudio() (string, error) {
	resp, err := http.DefaultClient.Do(mustRequest(audioURL))
	if err != nil {
		return "", fmt.Errorf("failed to fetch audio: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	return base64.StdEncoding.EncodeToString(body), nil
}
