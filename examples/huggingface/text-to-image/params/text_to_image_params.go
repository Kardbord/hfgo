package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const outputFile = "text_to_image_params.png"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Create a new client with your API token and desired model.
	// Generated images are binary and can exceed the 1 MiB default response
	// cap, so raise it to leave headroom.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("stabilityai/stable-diffusion-3-medium-diffusers"),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	const prompt = "A majestic lion in the savanna"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: prompt,
		Parameters: &hftypes.TextToImageParameters{
			GuidanceScale:     new(7.5),
			NegativePrompt:    new("blurry, low quality"),
			NumInferenceSteps: new(30),
			Width:             new(512),
			Height:            new(512),
			Seed:              new(int64(42)),
		},
	})
	if err != nil {
		log.Fatalf("error generating image: %v\n", err)
	}

	if err := os.WriteFile(outputFile, image.Image, 0o600); err != nil {
		log.Fatalf("error writing image: %v\n", err)
	}

	fmt.Printf("Wrote %d bytes of %s to %s\n", len(image.Image), image.MediaType, outputFile)
}
