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

	// Build a static model resolver that maps a Hugging Face model ID to the
	// WaveSpeed provider's model identifier for the text-to-video task.
	// For live Hub mappings, use hfproviders.NewHubModelResolver instead.
	mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskTextToVideo,
			Model:    "ZhengmingYu/DMAD",
		}: "wavespeed-ai/minimax-h3/text-to-video-lora",
	})

	// Create a WavespeedProvider backed by the resolver. The constructor
	// returns an error if the resolver is nil.
	provider, err := hfproviders.NewWavespeedProvider(mapper)
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

	video, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: prompt,
		Parameters: &hftypes.TextToVideoParameters{
			NumFrames: new(24),
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
