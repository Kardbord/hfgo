package hfgo

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/task"
)

// Client represents a Hugging Face API client with configured [hfopts.Options].
// Client instances are immutable; options are fixed at creation time and never mutated.
// This keeps client usage safe across goroutines and avoids surprises from mutable state.
// If options include externally-owned pointers, callers must avoid mutating them after creation
// or ensure their own synchronization.
type Client struct {
	opts hfopts.Options
}

// NewClient creates a new [Client] instance with the provided request options.
// If no options are provided, default options will be used.
// Clients are immutable in order to provide concurrency guarantees.
// To change options, create a new [Client] or provide per-call [hfopts.Options].
func NewClient(opts ...hfopts.Option) Client {
	return Client{
		opts: hfopts.NewOptions().With(opts...),
	}
}

// Chat sends a chat completion request and returns a [hftypes.ChatResponse].
//
// The request is passed by value and the SDK never mutates the received
// payload. The value copy shares the request's nested data (slices, maps, and
// pointed-to values) with the caller, so the caller must treat the request and
// the data it references as read-only while a call is in flight.
//
// Concurrency:
//   - A single Client is safe for concurrent use.
//   - Reusing one request across sequential, fully-awaited calls is safe.
//   - To invoke the same template request from multiple goroutines, pass a
//     defensive copy per call, e.g. go client.Chat(req.Clone(), ...), or build a
//     fresh request per call.
//
// Model Precedence:
// The Model field is resolved with the following precedence (highest to lowest):
//  1. ChatRequest.Model field (if non-nil and non-empty)
//  2. Per-request options Model override
//  3. Client-level Model option
//
// Provider Precedence:
// The Provider option is used as a suffix appended to the model string for
// routing in OpenAI-compatible endpoints. If the Model is in the format
// "model:provider", the Provider option is ignored. When the provider is
// the default Hugging Face provider, no suffix is appended and the router
// selects a provider automatically.
//
// For example:
//   - Model="mistral-7b", Provider=HuggingFaceProvider → "mistral-7b"
//   - Model="mistral-7b", Provider=SomeProvider → "mistral-7b:someprovider"
//   - Model="mistral-7b:sambanova", Provider=HuggingFaceProvider → "mistral-7b:sambanova" (Provider ignored)
//
// Provider selection is otherwise delegated to the HF router. To select a
// provider or selection policy explicitly, append a suffix to the model
// string (e.g. "model:sambanova", "model:fastest", "model:cheapest",
// "model:preferred"). See the Inference Providers docs for the default
// (no-suffix) behavior:
// https://huggingface.co/docs/inference-providers/main/en/index.
//
// Behavior:
//   - Returns a configuration error if the request is missing a model or messages.
//   - Returns a configuration error if *req.Stream is true; use [Client.ChatStream] for streaming.
//
//nolint:gocritic // hugeParam: Chat takes the request by value so the SDK never mutates the caller's payload
func (c Client) Chat(req hftypes.ChatRequest, opts ...hfopts.Option) (hftypes.ChatResponse, error) {
	return task.Chat(c.opts.With(opts...), req)
}

// ChatStream sends a chat completion request and returns a streaming [hftypes.ChatStream].
// Callers should Close the returned [hftypes.ChatStream] when finished so the underlying HTTP
// connection and decoder goroutine are released promptly.
//
// The request is passed by value and the SDK never mutates the received
// payload. The value copy shares the request's nested data (slices, maps, and
// pointed-to values) with the caller, so the caller must treat the request and
// the data it references as read-only while a call is in flight.
//
// Concurrency:
//   - A single Client is safe for concurrent use.
//   - Reusing one request across sequential, fully-awaited calls is safe.
//   - To invoke the same template request from multiple goroutines, pass a
//     defensive copy per call, e.g. go client.ChatStream(req.Clone(), ...), or
//     build a fresh request per call.
//
// Model Precedence:
// The Model field is resolved with the following precedence (highest to lowest):
//  1. ChatRequest.Model field (if non-nil and non-empty)
//  2. Per-request options Model override
//  3. Client-level Model option
//
// Provider Precedence:
// The Provider option is used as a suffix appended to the model string for
// routing in OpenAI-compatible endpoints. If the Model is in the format
// "model:provider", the Provider option is ignored. When the provider is
// the default Hugging Face provider, no suffix is appended and the router
// selects a provider automatically.
//
// For example:
//   - Model="mistral-7b", Provider=HuggingFaceProvider → "mistral-7b"
//   - Model="mistral-7b", Provider=SomeProvider → "mistral-7b:someprovider"
//   - Model="mistral-7b:sambanova", Provider=HuggingFaceProvider → "mistral-7b:sambanova" (Provider ignored)
//
// Provider selection is otherwise delegated to the HF router. To select a
// provider or selection policy explicitly, append a suffix to the model
// string (e.g. "model:sambanova", "model:fastest", "model:cheapest",
// "model:preferred"). See the Inference Providers docs for the default
// (no-suffix) behavior:
// https://huggingface.co/docs/inference-providers/main/en/index.
//
// Behavior:
//   - Returns a configuration error if the request is missing a model or messages.
//   - Always sends the request with streaming enabled.
//
//nolint:gocritic // hugeParam: ChatStream takes the request by value so the SDK never mutates the caller's payload
func (c Client) ChatStream(
	req hftypes.ChatRequest,
	opts ...hfopts.Option,
) (*hftypes.ChatStream, error) {
	return task.ChatStream(c.opts.With(opts...), req)
}

// ClassifyText sends a text classification request and returns the text
// classification response for a single input.
func (c Client) ClassifyText(
	req hftypes.TextClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.TextClassification, error) {
	return task.ClassifyText(c.opts.With(opts...), req)
}

// ClassifyTokens sends a token classification request and returns the token
// classification response for a single input.
func (c Client) ClassifyTokens(
	req hftypes.TokenClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.TokenClassification, error) {
	return task.ClassifyTokens(c.opts.With(opts...), req)
}

// ClassifyImage sends an image classification request and returns a list of
// image classification predictions for the single input, ordered by score
// (descending).
func (c Client) ClassifyImage(
	req hftypes.ImageClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.ImageClassification, error) {
	return task.ClassifyImage(c.opts.With(opts...), req)
}

// AnswerQuestion sends a question answering request and returns the answers.
//
// The request must include both a question and a context. The model will
// identify the answer to the question within the provided context.
func (c Client) AnswerQuestion(
	req hftypes.QuestionAnsweringRequest,
	opts ...hfopts.Option,
) ([]hftypes.QuestionAnswering, error) {
	return task.AnswerQuestion(c.opts.With(opts...), req)
}

// ZeroShotClassifyText sends a zero-shot text classification request and
// returns the zero-shot text classification response for a single input.
func (c Client) ZeroShotClassifyText(
	req hftypes.ZeroShotTextClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.ZeroShotTextClassification, error) {
	return task.ZeroShotClassifyText(c.opts.With(opts...), req)
}

// FillMask sends a fill mask request and returns the mask filling predictions
// for a single input.
func (c Client) FillMask(
	req hftypes.FillMaskRequest,
	opts ...hfopts.Option,
) ([]hftypes.FillMaskPrediction, error) {
	return task.FillMask(c.opts.With(opts...), req)
}

// Summarize sends a summarization request and returns the summarization output
// for a single input.
//
// The API always returns a list for summarization; a single input yields a
// one-element list rather than a bare summary object.
func (c Client) Summarize(
	req hftypes.SummarizationRequest,
	opts ...hfopts.Option,
) ([]hftypes.Summarization, error) {
	return task.Summarize(c.opts.With(opts...), req)
}

// AnswerTableQuestion sends a table question answering request and returns the answer.
//
// The request must include both a question and a table. The model will
// identify the answer to the question within the provided table data.
//
// NOTE: The Hugging Face API returns a bare JSON object for table question
// answering, not an array — despite the upstream schema declaring an array
// response. This method returns a single TableQuestionAnswer to match the
// actual API behavior.
func (c Client) AnswerTableQuestion(
	req hftypes.TableQuestionAnsweringRequest,
	opts ...hfopts.Option,
) (hftypes.TableQuestionAnswer, error) {
	return task.AnswerTableQuestion(c.opts.With(opts...), req)
}

// Translate sends a translation request and returns the translation output
// for a single input.
//
// The API always returns a list for translation; a single input yields a
// one-element list rather than a bare translation object.
func (c Client) Translate(
	req hftypes.TranslationRequest,
	opts ...hfopts.Option,
) ([]hftypes.Translation, error) {
	return task.Translate(c.opts.With(opts...), req)
}

// ExtractFeatures sends a feature extraction request and returns the pooled
// embedding vector for a single input.
//
// This method is for sentence-transformers-style models that reduce the output
// to one vector for the whole input. For transformers-style models that return
// one vector per token, use ExtractTokenFeatures. For multiple inputs, use
// ExtractFeaturesBatch.
func (c Client) ExtractFeatures(
	req hftypes.FeatureExtractionRequest,
	opts ...hfopts.Option,
) (hftypes.FeatureExtraction, error) {
	return task.ExtractFeatures(c.opts.With(opts...), req)
}

// ExtractFeaturesBatch sends a feature extraction request for a batch of inputs
// and returns pooled embedding vectors for each input in the batch.
//
// This method is for sentence-transformers-style models that reduce each input
// to one vector. For transformers-style models that return one vector per token,
// use ExtractTokenFeaturesBatch. Batch input is part of the upstream inference
// schema: the API accepts either a string or a list of strings.
func (c Client) ExtractFeaturesBatch(
	req hftypes.FeatureExtractionBatchRequest,
	opts ...hfopts.Option,
) ([]hftypes.FeatureExtraction, error) {
	return task.ExtractFeaturesBatch(c.opts.With(opts...), req)
}

// ExtractTokenFeatures sends a feature extraction request and returns one
// embedding vector per token for a single input.
//
// Use this with transformers-style models such as BART that return per-token
// hidden states. The outer slice corresponds to tokens; each inner slice is the
// hidden-state vector for that token. For pooled sentence embeddings, use
// ExtractFeatures.
func (c Client) ExtractTokenFeatures(
	req hftypes.FeatureExtractionRequest,
	opts ...hfopts.Option,
) (hftypes.FeatureExtractionTokens, error) {
	return task.ExtractTokenFeatures(c.opts.With(opts...), req)
}

// ExtractTokenFeaturesBatch sends a feature extraction request for a batch of
// inputs and returns token-level embeddings for each input.
//
// Use this with transformers-style models that return per-token hidden states.
// The outer slice corresponds to inputs, the middle slice to tokens within an
// input, and the inner slice to the hidden-state vector for a token. For pooled
// batch embeddings, use ExtractFeaturesBatch.
func (c Client) ExtractTokenFeaturesBatch(
	req hftypes.FeatureExtractionBatchRequest,
	opts ...hfopts.Option,
) (hftypes.FeatureExtractionBatchTokens, error) {
	return task.ExtractTokenFeaturesBatch(c.opts.With(opts...), req)
}

// DetectObjects sends an object detection request and returns a list
// of object detections.
func (c Client) DetectObjects(
	req hftypes.ObjectDetectionRequest,
	opts ...hfopts.Option,
) ([]hftypes.ObjectDetection, error) {
	return task.DetectObjects(c.opts.With(opts...), req)
}

// GenerateImage sends a text-to-image request and returns the generated image
// together with the media type the API advertised for it (for example,
// "image/png"). The image bytes are available on the response's Image field.
//
// Callers that only need to sniff the format can inspect the bytes with
// net/http.DetectContentType or, when it matters, pin the desired format via
// the model/provider.
//
// Generated images are binary and can exceed the default 1 MiB response cap, in
// which case the request fails with a configuration error. Use
// hfopts.WithMaxResponseBodyBytes to raise the cap for larger outputs.
func (c Client) GenerateImage(
	req hftypes.TextToImageRequest,
	opts ...hfopts.Option,
) (hftypes.TextToImageResponse, error) {
	return task.GenerateImage(c.opts.With(opts...), req)
}

// SegmentImage sends an image segmentation request and returns a list of
// predicted masks / segments.
func (c Client) SegmentImage(
	req hftypes.ImageSegmentationRequest,
	opts ...hfopts.Option,
) ([]hftypes.ImageSegmentation, error) {
	return task.SegmentImage(c.opts.With(opts...), req)
}

// RecognizeSpeech sends a speech recognition (automatic speech recognition)
// request and returns the recognized text, optionally with chunk-level
// timestamps.
//
// The request's Input field must be a base64-encoded audio payload. When no
// Parameters are provided, the audio is sent as a raw byte payload with the
// Content-Type auto-detected from the decoded bytes (via magic-byte sniffing).
// When Parameters are present, the request is sent as JSON with the
// base64-encoded audio in the inputs field.
//
// To override the auto-detected Content-Type for the raw path, set it via
// hfopts.WithHeader, e.g. hfopts.WithHeader("Content-Type", "audio/wav").
// Caller-supplied headers always take precedence over codec defaults.
func (c Client) RecognizeSpeech(
	req hftypes.SpeechRecognitionRequest,
	opts ...hfopts.Option,
) (hftypes.SpeechRecognitionResponse, error) {
	return task.RecognizeSpeech(c.opts.With(opts...), req)
}
