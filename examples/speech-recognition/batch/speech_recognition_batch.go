package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
)

// TODO: Test me
func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	client := hfgo.NewClient(
		hfgo.WithToken(token),
		hfgo.WithModel("openai/whisper-large-v3"),
	)

	inputs := []string{
		"UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
		"UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
	}

	fmt.Println("Recognizing speech (batch):")
	PrintJSON(inputs)
	fmt.Println("...")

	results, err := client.RecognizeSpeechBatch(
		hfgo.SpeechRecognitionBatchRequest{
			Inputs: inputs,
		},
	)
	if err != nil {
		log.Fatalf("error running batched speech recognition: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(results)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
