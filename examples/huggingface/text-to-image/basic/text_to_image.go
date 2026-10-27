package main

import (
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const outputFile = "text_to_image.png"

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

	const prompt = "A serene mountain landscape at sunset"
	fmt.Println("Generating image for prompt: " + prompt)
	fmt.Println("...")

	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: prompt,
	})
	if err != nil {
		log.Fatalf("error generating image: %v\n", err)
	}

	// The response carries the raw image data, so write it straight to disk.
	if err := os.WriteFile(outputFile, image.Image, 0o600); err != nil {
		log.Fatalf("error writing image: %v\n", err)
	}

	fmt.Printf("Wrote %d bytes of %s to %s\n", len(image.Image), image.MediaType, outputFile)
}
