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

const outputFile = "wavespeed_image_to_image.png"

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
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(token),
	)

	// Create a WavespeedProvider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewWavespeedProvider(resolver)
	if err != nil {
		log.Fatalf("error creating Wavespeed provider: %v", err)
	}

	// Configure the client to route image-to-image requests through the
	// WaveSpeed provider. Generated images are binary and can exceed the
	// default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("black-forest-labs/FLUX.2-klein-base-4B"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	// The input is a base64-encoded image. Replace with your own image data.
	const inputImage = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	const prompt = "Make the image blue"

	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	image, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: inputImage,
		Parameters: &hftypes.ImageToImageParameters{
			Prompt: new(prompt),
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
}
