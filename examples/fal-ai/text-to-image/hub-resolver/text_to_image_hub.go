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

const outputFile = "falai_text_to_image.png"

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

	// Configure the client to route text-to-image requests through the
	// Fal AI provider. Generated images are binary and can exceed the
	// default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("krea/Krea-2-Turbo"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "A serene mountain landscape at sunset"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	// fal's own request body is flat and provider-specific: the SDK encodes
	// the canonical request into it, runs fal's queue exchange (submit,
	// poll, fetch), and downloads the generated image. Provider-specific
	// request parameters that the canonical request types do not cover ride
	// on ProviderParams.
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

	// The same provider also serves image-to-image, text-to-video,
	// image-segmentation, and speech-recognition tasks; see the other
	// examples under examples/fal-ai.
}
