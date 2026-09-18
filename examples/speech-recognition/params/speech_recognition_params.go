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

	input := "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="

	fmt.Println("Recognizing speech with parameters:")
	PrintJSON(input)
	fmt.Println("...")

	result, err := client.RecognizeSpeech(
		hfgo.SpeechRecognitionRequest{
			Input: input,
			Parameters: &hfgo.SpeechRecognitionRequestParameters{
				ReturnTimestamps: ptr(true),
				GenerationParameters: &hfgo.SpeechRecognitionGenerationParameters{
					Temperature:   ptr(0.0),
					MaxNewTokens:  ptr(100),
					EarlyStopping: ptr(hfgo.EarlyStoppingNever),
				},
			},
		},
	)
	if err != nil {
		log.Fatalf("error running speech recognition: %v\n", err)
	}

	fmt.Println("Result:")
	PrintJSON(result)
}

func ptr[T any](v T) *T {
	return &v
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
