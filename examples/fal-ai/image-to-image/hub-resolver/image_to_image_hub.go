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

const outputFile = "falai_image_to_image.png"

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

	// Configure the client to route image-to-image requests through the
	// Fal AI provider. Generated images are binary and can exceed the
	// default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("black-forest-labs/FLUX.2-klein-9B"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	// fal accepts an HTTP(S) image URL as input and passes it through
	// unchanged; raw base64 is wrapped as a data: URI automatically.
	const inputImage = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"
	const prompt = "Turn the glasses into a tiger"

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
