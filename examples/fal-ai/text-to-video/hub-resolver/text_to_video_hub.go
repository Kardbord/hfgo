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

const outputFile = "falai_text_to_video.mp4"

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

	// Create a FalAIProvider backed by the resolver. The constructor returns
	// an error if the resolver is nil.
	provider, err := hfproviders.NewFalAIProvider(resolver)
	if err != nil {
		log.Fatalf("error creating Fal AI provider: %v", err)
	}

	// Configure the client to route text-to-video requests through the
	// Fal AI provider. Generated videos are binary and far exceed the
	// default 1 MiB response cap, so raise it generously. Video generation
	// can also take several minutes: the queue exchange is bounded by the
	// request context, so pass a deadline with hfopts.WithContext when the
	// default is too tight.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("Wan-AI/Wan2.2-TI2V-5B"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(64<<20),
	)

	const prompt = "A young man walking on the street"
	fmt.Println("Generating video for prompt: " + prompt)
	fmt.Println("...")

	// The SDK runs fal's queue exchange (submit, poll, fetch) and downloads
	// the generated video. Provider-specific request parameters that the
	// canonical request types do not cover ride on ProviderParams.
	video, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: prompt,
	})
	if err != nil {
		log.Fatalf("error generating video: %v", err)
	}

	// The response carries the raw video data, so write it straight to disk.
	if err := os.WriteFile(outputFile, video.Video, 0o600); err != nil {
		log.Fatalf("error writing video: %v", err)
	}

	fmt.Printf("Wrote %d bytes of %s to %s\n", len(video.Video), video.MediaType, outputFile)
}
