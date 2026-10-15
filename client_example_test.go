package hfgo_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// testClient returns a factory that produces an HTTP client with a dedicated
// transport. This keeps examples isolated from environment proxy settings and
// pointed at the local httptest.Server.
func testClient() func() http.Client {
	return func() http.Client {
		//nolint:exhaustruct // Zero-value Transport uses no proxy and HTTP/1.1.
		return http.Client{Transport: &http.Transport{}}
	}
}

// newMockJSONServer starts a local httptest.Server that always responds with
// 200 OK and Content-Type: application/json, writing body to every request.
// The response body is hard-coded so the example can run without a real token;
// copying this into the Go Playground will still hit the mock, not Hugging Face.
func newMockJSONServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
}

// newMockEventStreamServer starts a local httptest.Server that always responds
// with 200 OK and Content-Type: text/event-stream, writing body to every
// request and flushing it. Like newMockJSONServer, the body is hard-coded.
func newMockEventStreamServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("ResponseWriter does not implement http.Flusher")
		}
		fmt.Fprint(w, body)
		flusher.Flush()
	}))
}

// newMockImageServer starts a small httptest.Server that always responds with
// 200 OK, Content-Type: image/png, and an 8-byte PNG signature. The bytes are
// hard-coded so the example can run without a real token.
func newMockImageServer() *httptest.Server {
	png := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(png)
	}))
}

// tinyPNGBase64 is a 1x1 transparent PNG encoded as base64 for examples that
// need an image input but should not depend on external files.
const tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1UAAAAASUVORK5CYII="

// Example demonstrates how to create a Client and perform a chat completion.
func Example() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `{"id":"chatcmpl-1","created":1,"model":"test-model",` +
		`"system_fingerprint":"fp_1","choices":[` +
		`{"finish_reason":"stop","index":0,"message":` +
		`{"role":"assistant","content":"The capital of France is Paris."}}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	response, err := client.Chat(hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: new("What is the capital of France?"),
				},
			},
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(*response.Choices[0].Message.Content)
	// Output: The capital of France is Paris.
}

// ExampleClient_Chat_withOptions demonstrates setting per-request options such
// as temperature and max tokens via the request DTO.
func ExampleClient_Chat_withOptions() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `{"id":"chatcmpl-2","created":1,"model":"test-model",` +
		`"system_fingerprint":"fp_2","choices":[` +
		`{"finish_reason":"stop","index":0,"message":` +
		`{"role":"assistant","content":"Hi there!"}}],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	temperature := 0.7
	maxTokens := 256
	response, err := client.Chat(
		hftypes.ChatRequest{
			Messages: []hftypes.ChatMessage{
				{
					Role: "user",
					Content: hftypes.ChatMessageContent{
						Text: new("Hello!"),
					},
				},
			},
			Model:       new("per-request-model"),
			Temperature: &temperature,
			MaxTokens:   &maxTokens,
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(*response.Choices[0].Message.Content)
	// Output: Hi there!
}

// ExampleClient_ChatStream demonstrates streaming chat completions.
// Always close the stream when finished.
func ExampleClient_ChatStream() {
	const eventBody = "data: {" +
		`"id":"chatcmpl-3","created":1,"model":"test-model",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]` +
		"}\n\n" +
		"data: {" +
		`"id":"chatcmpl-3","created":1,"model":"test-model",` +
		`"choices":[{"index":0,"delta":{"content":", world!"}}]` +
		"}\n\n" +
		"data: [DONE]\n\n"

	server := newMockEventStreamServer(eventBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	stream, err := client.ChatStream(hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: new("Say hello"),
				},
			},
		},
	})
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	var result strings.Builder
	for {
		chunk, err := stream.Recv(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != nil {
				result.WriteString(*choice.Delta.Content)
			}
		}
	}

	fmt.Println(result.String())
	// Output: Hello, world!
}

// ExampleClient_ClassifyText demonstrates text classification.
func ExampleClient_ClassifyText() {
	server := newMockJSONServer(`[[{"label":"positive","score":0.95}]]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.ClassifyText(hftypes.TextClassificationRequest{
		Input: "I love this product!",
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (%.2f)\n", result[0].Label, result[0].Score)
	// Output: positive (0.95)
}

// ExampleClient_ClassifyTokens demonstrates token (named entity) classification.
func ExampleClient_ClassifyTokens() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.ClassifyTokens(hftypes.TokenClassificationRequest{
		Input: "My name is Sarah.",
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s: %s\n", *result[0].Entity, result[0].Word)
	// Output: PER: Sarah
}

// ExampleClient_ZeroShotClassifyText demonstrates zero-shot text classification.
func ExampleClient_ZeroShotClassifyText() {
	server := newMockJSONServer(`[{"label":"positive","score":0.95}]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.ZeroShotClassifyText(hftypes.ZeroShotTextClassificationRequest{
		Input: "This is a great product!",
		Parameters: &hftypes.ZeroShotTextClassificationParameters{
			CandidateLabels: []string{"positive", "negative", "neutral"},
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (%.2f)\n", result[0].Label, result[0].Score)
	// Output: positive (0.95)
}

// ExampleClient_FillMask demonstrates masked language modeling.
func ExampleClient_FillMask() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `[{"sequence":"The capital of France is Paris.","score":0.95,` +
		`"token":1,"token_str":"Paris"}]`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.FillMask(hftypes.FillMaskRequest{
		Input: "The capital of France is [MASK].",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(*result[0].TokenStr)
	// Output: Paris
}

// ExampleClient_Summarize demonstrates text summarization.
func ExampleClient_Summarize() {
	server := newMockJSONServer(`[{"summary_text":"A concise summary."}]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.Summarize(hftypes.SummarizationRequest{
		Input: "Some long text that should be summarized.",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result[0].SummaryText)
	// Output: A concise summary.
}

// ExampleClient_Translate demonstrates text translation.
func ExampleClient_Translate() {
	server := newMockJSONServer(`[{"translation_text":"Bonjour le monde."}]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.Translate(hftypes.TranslationRequest{
		Input: "Hello world.",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result[0].TranslationText)
	// Output: Bonjour le monde.
}

// ExampleClient_Translate_withOptions demonstrates translation with source and
// target language parameters.
func ExampleClient_Translate_withOptions() {
	server := newMockJSONServer(`[{"translation_text":"Bonjour le monde."}]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	srcLang := "en"
	tgtLang := "fr"
	result, err := client.Translate(
		hftypes.TranslationRequest{
			Input: "Hello world.",
			Parameters: &hftypes.TranslationParameters{
				SrcLang: &srcLang,
				TgtLang: &tgtLang,
			},
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(result[0].TranslationText)
	// Output: Bonjour le monde.
}

// ExampleClient_AnswerQuestion demonstrates extractive question answering.
func ExampleClient_AnswerQuestion() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `[{"answer":"Paris","score":0.95,"start":48,"end":53}]`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.AnswerQuestion(hftypes.QuestionAnsweringRequest{
		Input: hftypes.QuestionAnsweringInput{
			Question: "What is the capital of France?",
			Context:  "France is a country in Europe. Its capital is Paris.",
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result[0].Answer)
	// Output: Paris
}

// ExampleClient_AnswerTableQuestion demonstrates table-based question answering.
func ExampleClient_AnswerTableQuestion() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `{"answer":"30","cells":["30"],"coordinates":[[1,1]]}`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.AnswerTableQuestion(hftypes.TableQuestionAnsweringRequest{
		Input: hftypes.TableQuestionAnsweringInput{
			Question: "How old is Bob?",
			Table: map[string][]string{
				"Name": {"Alice", "Bob", "Carol"},
				"Age":  {"25", "30", "35"},
			},
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Answer)
	// Output: 30
}

// ExampleClient_ExtractFeatures demonstrates feature extraction (embeddings).
func ExampleClient_ExtractFeatures() {
	server := newMockJSONServer("[0.1, 0.2, 0.3]")
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	embedding, err := client.ExtractFeatures(hftypes.FeatureExtractionRequest{
		Input: "Hello world",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(embedding)
	// Output: [0.1 0.2 0.3]
}

// ExampleClient_ExtractFeaturesBatch demonstrates batched feature extraction.
func ExampleClient_ExtractFeaturesBatch() {
	server := newMockJSONServer("[[0.1,0.2,0.3],[0.4,0.5,0.6]]")
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	embeddings, err := client.ExtractFeaturesBatch(hftypes.FeatureExtractionBatchRequest{
		Inputs: []string{"hello", "world"},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(embeddings[0], embeddings[1])
	// Output: [0.1 0.2 0.3] [0.4 0.5 0.6]
}

// ExampleClient_ClassifyImage demonstrates image classification.
func ExampleClient_ClassifyImage() {
	server := newMockJSONServer(`[{"label":"Egyptian cat","score":0.514}]`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: tinyPNGBase64,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (%.3f)\n", result[0].Label, result[0].Score)
	// Output: Egyptian cat (0.514)
}

// ExampleClient_DetectObjects demonstrates object detection.
func ExampleClient_DetectObjects() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `[{"label":"person","score":0.95,"box":{"xmin":100,"ymin":50,"xmax":200,"ymax":250}}]`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
		Input: tinyPNGBase64,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s at (%d,%d)-(%d,%d)\n",
		result[0].Label,
		result[0].Box.XMin, result[0].Box.YMin,
		result[0].Box.XMax, result[0].Box.YMax,
	)
	// Output: person at (100,50)-(200,250)
}

// ExampleClient_GenerateImage demonstrates text-to-image generation.
func ExampleClient_GenerateImage() {
	server := newMockImageServer()
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: "a serene mountain landscape at sunset",
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("generated %d bytes of %s\n", len(image.Image), image.MediaType)
	// Output: generated 8 bytes of image/png
}

// ExampleClient_SegmentImage demonstrates image segmentation.
func ExampleClient_SegmentImage() {
	// The response body is hard-coded so the example can run without a real token.
	const responseBody = `[{"label":"cat","mask":"bWFzazE=","score":0.93}]`

	server := newMockJSONServer(responseBody)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	result, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: tinyPNGBase64,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (%.2f)\n", result[0].Label, *result[0].Score)
	// Output: cat (0.93)
}

// ExampleClient_RecognizeSpeech demonstrates automatic speech recognition.
func ExampleClient_RecognizeSpeech() {
	server := newMockJSONServer(`{"text":"hello world"}`)
	defer server.Close()

	client := hfgo.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithModel("test-model"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	// In a real use case this would be base64-encoded audio bytes.
	const audioBase64 = "AAAA"

	result, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: audioBase64,
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Text)
	// Output: hello world
}
