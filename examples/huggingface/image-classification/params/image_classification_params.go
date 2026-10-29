package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const imageURL = "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA"

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	// Create a new client with your API token and desired model
	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("google/vit-base-patch16-224"),
	)

	fmt.Println("Classifying image: " + imageURL)
	fmt.Println("...")

	imgData, err := fetchImage()
	if err != nil {
		log.Fatal("Failed to retrieve image URL: " + err.Error())
	}

	predictions, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: imgData,
		Parameters: &hftypes.ImageClassificationParameters{
			FunctionToApply: new(hftypes.ImageClassificationFuncSoftmax),
			TopK:            new(5),
		},
	})
	if err != nil {
		log.Fatalf("error running image classification: %v\n", err)
	}

	fmt.Println("Predictions:")
	printJSON(predictions)
}

// mustRequest creates an HTTP GET request with context. It panics if request creation fails.
func mustRequest(url string) *http.Request {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, http.NoBody)
	if err != nil {
		panic(err)
	}

	return req
}

// fetchImage retrieves the image at imageURL and returns it as
// base64-encoded data.
func fetchImage() (string, error) {
	resp, err := http.DefaultClient.Do(mustRequest(imageURL))
	if err != nil {
		return "", fmt.Errorf("failed to fetch image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	encoded := base64.StdEncoding.EncodeToString(body)

	return encoded, nil
}

func printJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
