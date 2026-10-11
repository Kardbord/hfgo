package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
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

	// Create a new client with your API token and desired model
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("openai/whisper-large-v3"),
	)

	fmt.Println("Transcribing audio with timestamps: " + audioURL)
	fmt.Println("...")

	audioInput, err := fetchAudio()
	if err != nil {
		log.Fatal("Failed to retrieve audio: " + err.Error())
	}

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioInput,
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	})
	if err != nil {
		log.Fatalf("error running speech recognition: %v\n", err)
	}

	fmt.Println("Full transcript:")
	fmt.Println(resp.Text)

	if resp.Chunks != nil {
		fmt.Println("")
		fmt.Println("Segments with timestamps:")
		printJSON(resp.Chunks)
	}
}

// mustRequest creates an HTTP GET request with context. It panics if request creation fails.
func mustRequest(url string) *http.Request {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, http.NoBody)
	if err != nil {
		panic(err)
	}

	return req
}

// fetchAudio retrieves the audio at audioURL and returns it as
// base64-encoded data.
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

	encoded := base64.StdEncoding.EncodeToString(body)

	return encoded, nil
}

func printJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
