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
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Build a Hub model resolver, which queries the Hub's
	// inferenceProviderMapping data for the configured model ID instead of
	// using a pinned table. The token raises rate limits and is required for
	// gated models; results are cached per resolver instance.
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(token),
	)

	// Create a DeepInfra provider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewDeepInfraProvider(resolver)
	if err != nil {
		log.Fatalf("error creating DeepInfra provider: %v", err)
	}

	// Configure the client to route speech-recognition requests through the
	// DeepInfra provider.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithProvider(provider),
	)

	// DeepInfra's transcription endpoint takes raw audio bytes, so the
	// example downloads a public sample and base64-encodes it for the
	// canonical request input. The provider sniffs the audio's media type to
	// label the multipart body, so no Content-Type option is needed.
	const audioURL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

	audio, err := fetchAudio(audioURL)
	if err != nil {
		log.Fatalf("error downloading audio: %v", err)
	}

	fmt.Println("Transcribing audio: " + audioURL)
	fmt.Println("...")

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: base64.StdEncoding.EncodeToString(audio),
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	})
	if err != nil {
		log.Fatalf("error running speech recognition: %v", err)
	}

	fmt.Println("Recognized text:")
	fmt.Println(resp.Text)

	if len(resp.Chunks) > 0 {
		fmt.Println("Chunks:")
		for _, chunk := range resp.Chunks {
			if len(chunk.Timestamp) == 2 {
				fmt.Printf("  [%v, %v] %s\n", chunk.Timestamp[0], chunk.Timestamp[1], chunk.Text)
			} else {
				fmt.Printf("  %s\n", chunk.Text)
			}
		}
	}

	// Provider-specific response metadata (e.g. usage, duration, language)
	// rides resp.ProviderMetadata.
}

// fetchAudio downloads the audio at url with its own timeout.
func fetchAudio(url string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status downloading audio: %s", resp.Status)
	}

	return io.ReadAll(resp.Body)
}
