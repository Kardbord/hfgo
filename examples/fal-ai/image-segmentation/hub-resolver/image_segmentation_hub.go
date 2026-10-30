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

	// Configure the client to route image-segmentation requests through the
	// Fal AI provider. Masks are returned inline, so raise the response cap
	// past the default 1 MiB for larger images.
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("briaai/RMBG-2.0"),
		hfopts.WithProvider(provider),
		hfopts.WithMaxResponseBodyBytes(16<<20),
	)

	// fal accepts an HTTP(S) image URL as input and passes it through
	// unchanged; raw base64 is wrapped as a data: URI automatically.
	const imageURL = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"

	fmt.Println("Segmenting image: " + imageURL)
	fmt.Println("...")

	segments, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: imageURL,
	})
	if err != nil {
		log.Fatalf("error running image segmentation: %v", err)
	}

	// fal reports a single anonymous mask, so the canonical response carries
	// one segment labeled "mask" whose Mask field is the base64-encoded
	// black-and-white image.
	for _, segment := range segments {
		fmt.Printf("label=%q mask_bytes(base64)=%d\n", segment.Label, len(segment.Mask))
	}
}
