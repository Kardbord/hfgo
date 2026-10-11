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
	"time"

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

	audioInput, err := fetchAudio(audioURL)
	if err != nil {
		log.Fatalf("Failed to fetch audio: %v\n", err)
	}

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioInput,
	}, hfopts.WithHeader("Content-Type", "audio/flac"))
	if err != nil {
		log.Fatalf("error running speech recognition: %v\n", err)
	}

	fmt.Println("Recognized text:")
	fmt.Println(resp.Text)
}

// fetchAudio retrieves the audio at audioURL and returns it as base64-encoded data.
func fetchAudio(audioURL string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, audioURL, http.NoBody)
	if err != nil {
		return "", err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
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
