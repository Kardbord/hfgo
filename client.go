package hfgo

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/task"
)

// Client represents a HuggingFace API client with configured request options.
// Client instances are immutable; options are fixed at creation time and never mutated.
// This keeps client usage safe across goroutines and avoids surprises from mutable state.
// If options include externally-owned pointers, callers must avoid mutating them after creation
// or ensure their own synchronization.
type Client struct {
	opts hfopts.Options
}

// NewClient creates a new Client instance with the provided request options.
// If no options are provided, default options will be used.
// Clients are immutable; to change options, create a new Client to keep calls deterministic.
func NewClient(opts ...hfopts.Option) Client {
	return Client{
		opts: hfopts.NewOptions().With(opts...),
	}
}

// Chat sends a chat completion request and returns a chat completion response.
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
// the default HuggingFace provider, no suffix is appended and the router
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
//   - Returns a configuration error if *req.Stream is true; use ChatStream for streaming.
//
//nolint:gocritic // hugeParam: Chat takes the request by value so the SDK never mutates the caller's payload
func (c Client) Chat(req hftypes.ChatRequest, opts ...hfopts.Option) (hftypes.ChatResponse, error) {
	return task.Chat(c.opts.With(opts...), req)
}

// ChatStream sends a chat completion request and returns a streaming response.
// Callers should Close the returned ChatStream when finished so the underlying HTTP
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
// the default HuggingFace provider, no suffix is appended and the router
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
	return task.StreamChat(c.opts.With(opts...), req)
}

// ClassifyText sends a text classification request and returns the text
// classification response for a single input.
//
// For multiple classification inputs, use ClassifyTextBatch.
func (c Client) ClassifyText(
	req hftypes.TextClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.TextClassification, error) {
	return task.ClassifyText(c.opts.With(opts...), req)
}

// ClassifyTextBatch sends a text classification request for a batch of inputs
// and returns a list of text classification responses for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) ClassifyTextBatch(
	req hftypes.TextClassificationBatchRequest,
	opts ...hfopts.Option,
) ([][]hftypes.TextClassification, error) {
	return task.ClassifyTextBatch(c.opts.With(opts...), req)
}

// ClassifyTokens sends a token classification request and returns the token
// classification response for a single input.
//
// For multiple inputs, use ClassifyTokensBatch.
func (c Client) ClassifyTokens(
	req hftypes.TokenClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.TokenClassification, error) {
	return task.ClassifyTokens(c.opts.With(opts...), req)
}

// ClassifyTokensBatch sends a token classification request for a batch of inputs
// and returns a list of token classification responses for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) ClassifyTokensBatch(
	req hftypes.TokenClassificationBatchRequest,
	opts ...hfopts.Option,
) ([][]hftypes.TokenClassification, error) {
	return task.ClassifyTokensBatch(c.opts.With(opts...), req)
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
//
// For multiple inputs, use ZeroShotClassifyTextBatch.
func (c Client) ZeroShotClassifyText(
	req hftypes.ZeroShotTextClassificationRequest,
	opts ...hfopts.Option,
) ([]hftypes.ZeroShotTextClassification, error) {
	return task.ZeroShotClassifyText(c.opts.With(opts...), req)
}

// ZeroShotClassifyTextBatch sends a zero-shot text classification request for
// a batch of inputs and returns a list of zero-shot text classification
// responses for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) ZeroShotClassifyTextBatch(
	req hftypes.ZeroShotTextClassificationBatchRequest,
	opts ...hfopts.Option,
) ([][]hftypes.ZeroShotTextClassification, error) {
	return task.ZeroShotClassifyTextBatch(c.opts.With(opts...), req)
}

// FillMask sends a fill mask request and returns the mask filling predictions
// for a single input.
//
// For multiple inputs, use FillMaskBatch.
func (c Client) FillMask(
	req hftypes.FillMaskRequest,
	opts ...hfopts.Option,
) ([]hftypes.FillMaskPrediction, error) {
	return task.FillMask(c.opts.With(opts...), req)
}

// FillMaskBatch sends a fill mask request for a batch of inputs and returns a
// list of mask filling predictions for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) FillMaskBatch(
	req hftypes.FillMaskBatchRequest,
	opts ...hfopts.Option,
) ([][]hftypes.FillMaskPrediction, error) {
	return task.FillMaskBatch(c.opts.With(opts...), req)
}

// Summarize sends a summarization request and returns the summarization output
// for a single input.
//
// The API always returns a list for summarization; a single input yields a
// one-element list rather than a bare summary object.
//
// For multiple inputs, use SummarizeBatch.
func (c Client) Summarize(
	req hftypes.SummarizationRequest,
	opts ...hfopts.Option,
) ([]hftypes.Summarization, error) {
	return task.Summarize(c.opts.With(opts...), req)
}

// SummarizeBatch sends a summarization request for a batch of inputs and returns
// a flat list of summarization outputs, one for each input in the batch, in the
// same order as the inputs.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice. The response is
// a flat list (one summary per input) — not a nested list — consistent with
// how the API returns a list even for a single input.
func (c Client) SummarizeBatch(
	req hftypes.SummarizationBatchRequest,
	opts ...hfopts.Option,
) ([]hftypes.Summarization, error) {
	return task.SummarizeBatch(c.opts.With(opts...), req)
}

// AnswerTableQuestion sends a table question answering request and returns the answer.
//
// The request must include both a question and a table. The model will
// identify the answer to the question within the provided table data.
//
// NOTE: The HuggingFace API returns a bare JSON object for table question
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
//
// For multiple inputs, use TranslateBatch.
func (c Client) Translate(
	req hftypes.TranslationRequest,
	opts ...hfopts.Option,
) ([]hftypes.Translation, error) {
	return task.Translate(c.opts.With(opts...), req)
}

// TranslateBatch sends a translation request for a batch of inputs and returns
// a flat list of translation outputs, one for each input in the batch, in the
// same order as the inputs.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice. The response is
// a flat list (one translation per input) — not a nested list — consistent with
// how the API returns a list even for a single input.
func (c Client) TranslateBatch(
	req hftypes.TranslationBatchRequest,
	opts ...hfopts.Option,
) ([]hftypes.Translation, error) {
	return task.TranslateBatch(c.opts.With(opts...), req)
}

// FeatureExtract sends a feature extraction request and returns the embedding
// vector for a single input.
//
// For multiple inputs, use FeatureExtractBatch.
//
// NOTE: hf-inference is NOT the only supported provider, add support for other providers.
func (c Client) FeatureExtract(
	req hftypes.FeatureExtractionRequest,
	opts ...hfopts.Option,
) (hftypes.FeatureExtraction, error) {
	return task.ExtractFeatures(c.opts.With(opts...), req)
}

// FeatureExtractBatch sends a feature extraction request for a batch of inputs
// and returns embedding vectors for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice. The response
// is a flat list of embedding vectors in the same order as the inputs.
//
// NOTE: hf-inference is NOT the only supported provider, add support for other providers.
func (c Client) FeatureExtractBatch(
	req hftypes.FeatureExtractionBatchRequest,
	opts ...hfopts.Option,
) ([]hftypes.FeatureExtraction, error) {
	return task.ExtractFeaturesBatch(c.opts.With(opts...), req)
}

// DetectObjects sends an object detection request and returns a list
// of object detections.
func (c Client) DetectObjects(
	req hftypes.ObjectDetectionRequest,
	opts ...hfopts.Option,
) ([]hftypes.ObjectDetection, error) {
	return task.DetectObjects(c.opts.With(opts...), req)
}

// SegmentImage sends an image segmentation request and returns a list of
// predicted masks / segments.
func (c Client) SegmentImage(
	req hftypes.ImageSegmentationRequest,
	opts ...hfopts.Option,
) ([]hftypes.ImageSegmentation, error) {
	return task.SegmentImage(c.opts.With(opts...), req)
}
