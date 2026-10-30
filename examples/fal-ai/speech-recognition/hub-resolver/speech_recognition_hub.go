package main

import (
	"fmt"
	"log"
	"os"

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

	// Create a Fal AI provider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewFalAIProvider(resolver)
	if err != nil {
		log.Fatalf("error creating Fal AI provider: %v", err)
	}

	// Configure the client to route speech-recognition requests through the
	// Fal AI provider.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("openai/whisper-large-v3"),
		hfopts.WithProvider(provider),
	)

	// fal accepts an HTTP(S) audio URL as input and passes it through
	// unchanged; raw base64 is wrapped as a data: URI labeled audio/mpeg.
	const audioURL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"

	fmt.Println("Transcribing audio: " + audioURL)
	fmt.Println("...")

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioURL,
	})
	if err != nil {
		log.Fatalf("error running speech recognition: %v", err)
	}

	fmt.Println("Recognized text:")
	fmt.Println(resp.Text)

	// Set Parameters.ReturnTimestamps to request chunk-level timestamps;
	// they surface on resp.Chunks.
}
