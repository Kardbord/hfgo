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

const outputFile = "wavespeed_text_to_image.png"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Build a static model resolver that maps a Hugging Face model ID to the
	// WaveSpeed provider's model identifier for the text-to-image task.
	// For live Hub mappings, use hfproviders.NewHubModelResolver instead.
	mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskTextToImage,
			Model:    "black-forest-labs/FLUX.1-dev",
		}: "wavespeed-ai/flux-dev",
	})

	// Create a WavespeedProvider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewWavespeedProvider(mapper)
	if err != nil {
		log.Fatalf("error creating Wavespeed provider: %v", err)
	}

	// Configure the client to route text-to-image requests through the
	// WaveSpeed provider. Generated images are binary and can exceed the
	// default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("black-forest-labs/FLUX.1-dev"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "A serene mountain landscape at sunset"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	// WaveSpeed's own request body is flat and provider-specific: the SDK
	// encodes the canonical request into it automatically, polls the task
	// result URL until completion, and downloads the generated image.
	// Provider-specific request parameters that the canonical request types
	// do not cover ride on ProviderParams.
	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: prompt,
		ProviderParams: map[string]any{
			"enable_safety_checker": true,
		},
	})
	if err != nil {
		log.Fatalf("error generating image: %v", err)
	}

	// The response carries the raw image data, so write it straight to disk.
	if err := os.WriteFile(outputFile, image.Image, 0o600); err != nil {
		log.Fatalf("error writing image: %v", err)
	}

	fmt.Printf("Wrote %d bytes of %s to %s\n", len(image.Image), image.MediaType, outputFile)

	// The same provider also serves image-to-image and text-to-video tasks;
	// see the other examples under examples/wavespeed.
}
