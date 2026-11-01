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

const outputFile = "nscale_text_to_image.png"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Build a static model resolver that maps a Hugging Face model ID to the
	// Nscale provider's model identifier for the text-to-image task. The
	// mapping is pinned in code, so resolution performs no Hub lookup and
	// works without network access to the Hub. See ../hub-resolver for live
	// Hub mappings instead.
	mappings := map[hfproviders.StaticMapping]hfproviders.ModelMapping{
		{
			Provider: "nscale",
			Task:     hfproviders.TaskTextToImage,
			Model:    "black-forest-labs/FLUX.1-schnell",
		}: {ProviderModelID: "black-forest-labs/FLUX.1-schnell"},
	}
	mapper := hfproviders.NewStaticModelResolver(mappings)

	// Create an NscaleProvider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewNscaleProvider(mapper)
	if err != nil {
		log.Fatalf("error creating Nscale provider: %v", err)
	}

	// Configure the client to route text-to-image requests through the
	// Nscale provider. Generated images arrive base64-encoded and can exceed
	// the default 1 MiB response cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("black-forest-labs/FLUX.1-schnell"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "A serene mountain landscape at sunset"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	// Nscale's own request body is OpenAI-style and provider-specific: the
	// SDK encodes the canonical request into it and decodes the base64 image
	// from the response. Canonical parameters are forwarded as-is, mirroring
	// huggingface.js (width/height stay flat; num_inference_steps is not
	// dropped). Provider-specific request parameters that the canonical
	// request types do not cover ride on ProviderParams.
	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: prompt,
		ProviderParams: map[string]any{
			"n": 1,
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

	// The same provider also serves chat completion tasks; see the other
	// examples under examples/nscale.
}
