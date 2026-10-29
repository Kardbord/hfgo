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

const outputFile = "wavespeed_text_to_video.mp4"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Build a Hub model resolver, which queries the Hub's
	// inferenceProviderMapping data for the configured model ID instead of
	// using a pinned table. The token raises rate limits and is required for
	// gated models; results are cached per resolver instance. See
	// ../static-resolver for a pinned mapping that needs no Hub lookup.
	//
	// ZhengmingYu/DMAD is a LoRA adapter, and the Hub's mapping declares that
	// metadata (adapter: "lora"), so the SDK still injects the "loras" payload
	// entry pointing at the repository — there is no adapter metadata to pin
	// by hand here.
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(token),
	)

	// Create a WavespeedProvider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewWavespeedProvider(resolver)
	if err != nil {
		log.Fatalf("error creating Wavespeed provider: %v", err)
	}

	// Configure the client to route text-to-video requests through the
	// WaveSpeed provider. Generated videos are binary and can exceed the
	// default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("ZhengmingYu/DMAD"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(32<<20),
	)

	const prompt = "A cat playing with a ball of yarn"
	fmt.Println("Generating video for prompt: " + prompt)
	fmt.Println("...")

	// WaveSpeed models take provider-native parameters, not the canonical
	// Hugging Face ones: this model accepts duration (seconds), resolution,
	// and aspect_ratio, and rejects unknown fields with a 400 — so the
	// canonical NumFrames/GuidanceScale/etc. parameters must not be sent.
	// Anything the typed request types do not cover rides on ProviderParams.
	video, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: prompt,
		ProviderParams: map[string]any{
			"duration": 5,
		},
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
