package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// Client represents a HuggingFace API client with configured request options.
// Client instances are immutable; options are fixed at creation time and never mutated.
// This keeps client usage safe across goroutines and avoids surprises from mutable state.
// If options include externally-owned pointers, callers must avoid mutating them after creation
// or ensure their own synchronization.
// RawService captures a snapshot of these options when created.
type Client struct {
	opts request.Options
}

// NewClient creates a new Client instance with the provided request options.
// If no options are provided, default options will be used.
// Clients are immutable; to change options, create a new Client to keep calls deterministic.
func NewClient(opts ...Option) Client {
	return Client{
		opts: request.NewOptions().With(opts...),
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
//   - Model="mistral-7b", Provider=mockProvider("sambanova") → "mistral-7b:sambanova"
//   - Model="mistral-7b:sambanova", Provider=HuggingFaceProvider → "mistral-7b:sambanova" (Provider ignored)
//
// Behavior:
//   - Returns a configuration error if the request is missing a model or messages.
//   - Returns a configuration error if *req.Stream is true; use ChatStream for streaming.
//
//nolint:gocritic // hugeParam: Chat takes the request by value so the SDK never mutates the caller's payload
func (c Client) Chat(req ChatRequest, opts ...Option) (ChatResponse, error) {
	return newChatService(c.opts).complete(req, opts...)
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
//   - Model="mistral-7b", Provider=mockProvider("sambanova") → "mistral-7b:sambanova"
//   - Model="mistral-7b:sambanova", Provider=HuggingFaceProvider → "mistral-7b:sambanova" (Provider ignored)
//
// Behavior:
//   - Returns a configuration error if the request is missing a model or messages.
//   - Always sends the request with streaming enabled.
//
//nolint:gocritic // hugeParam: ChatStream takes the request by value so the SDK never mutates the caller's payload
func (c Client) ChatStream(req ChatRequest, opts ...Option) (*ChatStream, error) {
	return newChatService(c.opts).completeStream(req, opts...)
}

// ClassifyText sends a text classification request and returns the text
// classification response for a single input.
//
// For multiple classification inputs, use ClassifyTextBatch.
func (c Client) ClassifyText(
	req TextClassificationRequest,
	opts ...Option,
) ([]TextClassification, error) {
	return newTextClassificationService(c.opts).classify(req, opts...)
}

// ClassifyTextBatch sends a text classification request for a batch of inputs
// and returns a list of text classification responses for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) ClassifyTextBatch(
	req TextClassificationBatchRequest,
	opts ...Option,
) ([][]TextClassification, error) {
	return newTextClassificationService(c.opts).classifyBatch(req, opts...)
}

// ClassifyTokens sends a token classification request and returns the token
// classification response for a single input.
//
// For multiple inputs, use ClassifyTokensBatch.
func (c Client) ClassifyTokens(
	req TokenClassificationRequest,
	opts ...Option,
) ([]TokenClassification, error) {
	return newTokenClassificationService(c.opts).classify(req, opts...)
}

// ClassifyTokensBatch sends a token classification request for a batch of inputs
// and returns a list of token classification responses for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) ClassifyTokensBatch(
	req TokenClassificationBatchRequest,
	opts ...Option,
) ([][]TokenClassification, error) {
	return newTokenClassificationService(c.opts).classifyBatch(req, opts...)
}

// AnswerQuestion sends a question answering request and returns the answers.
//
// The request must include both a question and a context. The model will
// identify the answer to the question within the provided context.
func (c Client) AnswerQuestion(
	req QuestionAnsweringRequest,
	opts ...Option,
) ([]QuestionAnswering, error) {
	return newQuestionAnsweringService(c.opts).answer(req, opts...)
}

// ZeroShotClassifyText sends a zero-shot text classification request and
// returns the zero-shot text classification response for a single input.
//
// For multiple inputs, use ZeroShotClassifyTextBatch.
func (c Client) ZeroShotClassifyText(
	req ZeroShotTextClassificationRequest,
	opts ...Option,
) ([]ZeroShotTextClassification, error) {
	return newZeroShotTextClassificationService(c.opts).classify(req, opts...)
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
	req ZeroShotTextClassificationBatchRequest,
	opts ...Option,
) ([][]ZeroShotTextClassification, error) {
	return newZeroShotTextClassificationService(c.opts).classifyBatch(req, opts...)
}

// FillMask sends a fill mask request and returns the mask filling predictions
// for a single input.
//
// For multiple inputs, use FillMaskBatch.
func (c Client) FillMask(req FillMaskRequest, opts ...Option) ([]FillMaskPrediction, error) {
	return newFillMaskService(c.opts).fill(req, opts...)
}

// FillMaskBatch sends a fill mask request for a batch of inputs and returns a
// list of mask filling predictions for each input in the batch.
//
// NOTE: Batched inference is supported by the upstream API, but is not
// officially documented; behavior may change without notice.
//
// Callers should check the length of the response list before indexing.
func (c Client) FillMaskBatch(
	req FillMaskBatchRequest,
	opts ...Option,
) ([][]FillMaskPrediction, error) {
	return newFillMaskService(c.opts).fillBatch(req, opts...)
}

// Summarize sends a summarization request and returns the summarization output
// for a single input.
//
// The API always returns a list for summarization; a single input yields a
// one-element list rather than a bare summary object.
//
// For multiple inputs, use SummarizeBatch.
func (c Client) Summarize(req SummarizationRequest, opts ...Option) ([]Summarization, error) {
	return newSummarizationService(c.opts).summarize(req, opts...)
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
	req SummarizationBatchRequest,
	opts ...Option,
) ([]Summarization, error) {
	return newSummarizationService(c.opts).summarizeBatch(req, opts...)
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
	req TableQuestionAnsweringRequest,
	opts ...Option,
) (TableQuestionAnswer, error) {
	return newTableQuestionAnsweringService(c.opts).answer(req, opts...)
}

// Translate sends a translation request and returns the translation output
// for a single input.
//
// The API always returns a list for translation; a single input yields a
// one-element list rather than a bare translation object.
//
// For multiple inputs, use TranslateBatch.
func (c Client) Translate(req TranslationRequest, opts ...Option) ([]Translation, error) {
	return newTranslationService(c.opts).translate(req, opts...)
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
	req TranslationBatchRequest,
	opts ...Option,
) ([]Translation, error) {
	return newTranslationService(c.opts).translateBatch(req, opts...)
}

// FeatureExtract sends a feature extraction request and returns the embedding
// vector for a single input.
//
// For multiple inputs, use FeatureExtractBatch.
//
// NOTE: hf-inference is NOT the only supported provider, add support for other providers.
func (c Client) FeatureExtract(
	req FeatureExtractionRequest,
	opts ...Option,
) (FeatureExtraction, error) {
	return newFeatureExtractionService(c.opts).extract(req, opts...)
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
	req FeatureExtractionBatchRequest,
	opts ...Option,
) ([]FeatureExtraction, error) {
	return newFeatureExtractionService(c.opts).extractBatch(req, opts...)
}

// Raw returns the raw HTTP request service for this client. Unlike the other
// endpoints, which are exposed directly as Client methods, the raw path remains
// namespaced under RawService: it is the advanced escape hatch for endpoints the
// SDK does not otherwise cover, and its several method variants are easier to
// discover grouped together than splashed across the Client surface.
//
// RawService is immutable and captures a snapshot of the client options when
// created; it is lightweight, so prefer calling Raw() per use rather than
// retaining the value.
func (c Client) Raw() RawService {
	return newRawService(c.opts)
}
