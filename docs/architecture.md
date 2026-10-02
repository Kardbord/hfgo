# hfgo SDK Architecture Documentation

## Project Overview

**hfgo** is a production-quality Go SDK for the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers/tasks/index). It provides Go bindings to perform inference tasks for any supported model available in the Hugging Face Model Hub.

- **Module**: `github.com/Kardbord/hfgo/v4`
- **Go Version**: 1.25+
- **License**: MIT (Copyright 2021 Tanner Kvarfordt)
- **Goal**: Production-ready, follows best practices and idioms, maintains feature parity with upstream API
- **Repository**: https://github.com/Kardbord/hfgo

## Repository & Package Layout

The repository is organized as a multi-package Go module. The public surface
consists of the root `hfgo` package plus the explicit public sub-packages
`hftypes`, `hfopts`, `hferrors`, `hfproviders`, and `hfgoversion`. Internal
packages under `internal/` are implementation details and should only be
imported from within the module.

### Public Packages

| Package | Import path | Contents |
|---------|-------------|----------|
| `hfgo` | `github.com/Kardbord/hfgo/v4` | `Client`, `NewClient`, and every inference endpoint method |
| `hfopts` | `github.com/Kardbord/hfgo/v4/hfopts` | `Options`, `Option`, and all `With*` option helpers |
| `hftypes` | `github.com/Kardbord/hfgo/v4/hftypes` | Request/response Data Transfer Objects for every task |
| `hferrors` | `github.com/Kardbord/hfgo/v4/hferrors` | `APIError`, `SDKError`, and `SDKErrorKind` definitions |
| `hfproviders` | `github.com/Kardbord/hfgo/v4/hfproviders` | `Provider` interface, `HuggingFaceProvider`, built-in codec |
| `hfraw` | `github.com/Kardbord/hfgo/v4/hfraw` | Low-level `Client`/`Stream`/`Event` escape hatch for arbitrary HTTP/SSE |
| `hfgoversion` | `github.com/Kardbord/hfgo/v4/hfgoversion` | `Version` constant and `UserAgent()` helper |

### `hfgo` (root package)

| File | Contents |
|------|----------|
| `client.go` | `Client`, `NewClient`, and every inference endpoint method |
| `doc.go` | Package-level design notes |

### `hfopts`

| File | Contents |
|------|----------|
| `options.go` | `Options`, `Option`, all `With*` helpers, and `Validate` |
| `doc.go` | Package documentation |

### `hftypes`

| File | Contents |
|------|----------|
| `{task}.go` | One file per task defining that task's DTOs, e.g. `fill_mask.go`, `text_classification.go` |
| `chat_request.go`, `chat_response.go`, `chat_streaming.go`, `chat_common.go` | Chat DTOs and `ChatStream` |
| `clone.go` | Deep `Clone` method for every request DTO |

### `hfraw` (public package)

Provides the low-level `Client`, `Stream`, and `Event` escape hatch for raw
HTTP and SSE access to endpoints the SDK does not model type-safely. Built on
top of `internal/request` and `hfopts`; advanced callers import it explicitly.

### `hfproviders/` (public package)

Defines the inference-provider abstraction: the `Provider` interface, the
built-in `HuggingFaceProvider` with its embedded `DefaultCodec`, and the
`Task` constants naming every supported inference task.

### `internal/` (implementation detail)

None of these packages are part of the public API.

| Package | Contents |
|---------|----------|
| `internal/task` | Package-level task functions (`Chat`, `StreamChat`, `ClassifyText`, …) |
| `internal/chatstream` | `ChatStream` implementation and the `ToolCallAccumulator` helper |
| `internal/request` | HTTP plumbing, JSON decode helpers, and SSE parsing |
| `internal/utils` | Shared helpers used by `hfopts`, `internal/request`, and internal packages |
| `internal/testutils` | Shared test helpers (mock transports, trackers, pointers) |
| `internal/integration_tests` | Live-API integration tests (tagged `integration`) |

### Re-export Pattern

DTOs are defined once in the public `hftypes` package. The `hfproviders`
package can import them directly without creating an import cycle, and the
root `hfgo` package consumes both `hftypes` and `hfproviders` to expose the
type-safe Client API.

## Core Architecture

### Client-Centric Design Pattern

The SDK follows a strict immutability pattern for concurrency safety:

1. **Client**: Immutable value type that captures configuration at creation time
   - Options are fixed and never mutated
   - Safe for concurrent use across goroutines
   - Each method call snapshots the client's options, so calls are independent and deterministic

2. **Client Methods**: Every inference endpoint is called directly on the `Client`
   - `Chat` / `ChatStream`: Chat completions
    - `ClassifyText` / `ClassifyTextBatch`: Text classification
    - `AnswerQuestion`: Question answering
    - `ClassifyTokens` / `ClassifyTokensBatch`: Token classification (named entity recognition)
    - `ZeroShotClassifyText` / `ZeroShotClassifyTextBatch`: Zero-shot text classification
   - `FillMask` / `FillMaskBatch`: Mask filling
   - `Summarize` / `SummarizeBatch`: Summarization
    - `Translate` / `TranslateBatch`: Translation
    - `AnswerTableQuestion`: Table question answering
    - `FeatureExtract` / `FeatureExtractBatch`: Feature extraction (embeddings)
    - Per-domain task functions in internal/task are unexported implementation details; callers interact only with the Client
   - The root package exposes only typed inference endpoints; raw HTTP/SSE access lives in the separate `hfraw` package (see below).

3. **Per-Request Options**: Can override client defaults for single calls
   - Applied by value with defensive header copies
   - Contexts and HTTP clients are shared references

### Key Design Principles

From `doc.go` and README:

- **Immutability**: Clients are immutable; to change options, create a new Client
- **Concurrency**: Clients are safe for concurrent use by default
- **Feature Parity**: SDK favors upstream API feature parity; breaking changes possible as API evolves
- **DTOs**: Request/response types closely aligned to the HuggingFace API
- **Streaming**: Server-Sent Events (SSE) based streaming for chat completions
- **Error Handling**: Distinct APIError vs SDKError types with categorization
- **HTTP Client Injection**: Factory functions return fresh client values; avoid sharing mutable transports unless synchronized

### Concurrency Safety

This SDK is **safe for concurrent use out of the box**. No explicit synchronization is required when using a client from multiple goroutines.

**Concurrency Guarantees**:
- **Clients**: Fully concurrent-safe as immutable value types
- **Client method calls**: Each call snapshots the client's options, so per-request overrides and concurrent calls never interfere
- **Shared HTTP clients**: If you inject an HTTP client via `hfopts.WithHTTPClientFactory()`, ensure it's either thread-safe by design or properly synchronized externally

**How It Works**:
The SDK achieves concurrency safety through immutability:
1. Clients never mutate their options after creation
2. Each method call derives a fresh snapshot from the client's options, applying per-request overrides by value (defensive copies)
3. No shared mutable state between goroutines

**Example**:
```go
// Safe: Single immutable client used by multiple goroutines
client := hfgo.NewClient(hfopts.WithToken(token), hfopts.WithModel("mistral-7b"))

// Each goroutine passes its OWN defensive copy of the request, so the shared
// template is only ever read — never mutated while another call is in flight.
go func() {
    resp, err := client.Chat(req.Clone())
    // ...
}()

go func() {
    stream, err := client.ChatStream(req.Clone())
    // ...
}()
```

### Provider Abstraction

The SDK routes requests and transforms wire formats through a pluggable
provider layer, defined in the public `hfproviders` package. The HuggingFace
Inference API is the reference wire format, so the built-in provider is a
no-op transform; third-party providers can implement the same interface to
serve HF-format requests from their own endpoints.

#### The `Provider` Interface

```go
type Provider interface {
    Endpoint(task Task, model string) (string, error)
    ProviderSuffix() string
    EncodeRequest(task Task, hfBody []byte, hfContentType string) (
        providerBody []byte, providerContentType string, err error,
    )
    DecodeResponse(task Task, providerBody []byte, providerContentType string) (
        hfBody []byte, hfContentType string, err error,
    )
}
```

- `Endpoint` returns the API endpoint path for a task and model, or an error if
  the model is invalid or the task is unsupported.
- `ProviderSuffix` returns the routing suffix appended to model IDs on
  OpenAI-compatible endpoints. An empty string (the HuggingFace provider)
  appends nothing.
- `EncodeRequest` / `DecodeResponse` transform between HF-spec and
  provider-spec request/response bodies. Implementations must be safe for
  concurrent use.

#### Built-ins

- **`HuggingFaceProvider`**: embeds `DefaultCodec` (identity transforms) and
  appends no routing suffix. Construct via `NewHuggingFaceProvider()`.
- **`DefaultCodec`**: identity codec passing request/response bodies through
  unchanged. Embedded by `HuggingFaceProvider` and available for other
  providers that speak the same wire format for a task.
- **`hfproviders.Task`**: string constants naming supported inference tasks
  (e.g. `TaskChatCompletion`, `TaskTextClassification`,
  `TaskFeatureExtraction`, `TaskTextToImage`, …).

#### Provider Selection and Suffix Routing

- On OpenAI-compatible endpoints (e.g. chat completions), a provider or
  selection policy can be pinned by appending a suffix to the model string
  (e.g. `model:sambanova`, `model:fastest`, `model:cheapest`,
  `model:preferred`); a `HuggingFaceProvider`-like empty suffix means the HF
  router selects the provider. See
  https://huggingface.co/docs/inference-providers/main/en/index.
- `hfopts.WithProvider(p)` sets `Options.Provider`; `hfopts.WithDefaultProvider()` sets
  the default `HuggingFaceProvider`.

#### Dispatch Flow

`internal/task/invoke.go` implements the request lifecycle shared by every
task function:

1. `resolveModelDispatch` validates options (model set, provider non-nil) and
   resolves the endpoint via `Provider.Endpoint`.
2. `encodeRequest` marshals the typed request DTO to JSON, then applies
   `Provider.EncodeRequest` to produce the wire body and `Content-Type`.
3. `doJSONInference` / `doStreamingInference` send the request through the
   `internal/request` layer, then convert the response back with
   `Provider.DecodeResponse` before unmarshalling into the typed response DTO.
   Streaming additionally applies the same transform per SSE `data` chunk.

This keeps task functions (`internal/task`) agnostic to wire format: providers
own endpoint construction and any request/response translation.

## Error Handling

### APIError
Represents errors returned by the HuggingFace API. Available at `github.com/Kardbord/hfgo/v4/hferrors.APIError`.

**Fields**:
- `StatusCode`: HTTP status code
- `Message`: Human-readable error message
- `Body`: Response body as io.ReadCloser (caller responsible for closing)
- `RequestID`: X-Request-ID header value if available
- `Method`: HTTP method used
- `URL`: URL that was requested

**Helper Methods**:
- `IsClientError()`: Returns true for 4xx status codes
- `IsServerError()`: Returns true for 5xx status codes
- `IsAuthenticationError()`: Returns true for 401 Unauthorized
- `IsRateLimitError()`: Returns true for 429 Too Many Requests

**Type Assertion Pattern**:
```go
if apiErr, ok := err.(*hferrors.APIError); ok {
    if apiErr.IsAuthenticationError() {
        // Handle auth error
    }
}
```

### SDKError
Represents client-side SDK errors that occur before API response or during response unmarshaling.
Available at `github.com/Kardbord/hfgo/v4/hferrors.SDKError`.

**Fields**:
- `Kind`: Error category (SDKErrorKind)
- `Message`: Human-readable error message
- `Err`: Underlying error (if any)

**Error Kinds**:
- `SDKErrorKindValidation`: Validation error in API responses
- `SDKErrorKindConfiguration`: Invalid or missing configuration
- `SDKErrorKindSerialization`: Serialization/deserialization error
- `SDKErrorKindTransport`: Transport-layer failure
- `SDKErrorKindInternal`: Internal SDK error

**Type Assertion Pattern**:
```go
if sdkErr, ok := err.(*hferrors.SDKError); ok {
    fmt.Printf("Kind %s: %s\n", sdkErr.Kind, sdkErr.Message)
}
```

## Configuration Options

All options are functions that receive an `hfopts.Options` value and are
collected as `hfopts.Option`. They are applied to clients at construction
time and can be overridden per-request:

```go
client := hfgo.NewClient(hfopts.WithToken(token), hfopts.WithModel("mistral-7b"))

// Per-request override
resp, err := client.Chat(req, hfopts.WithModel("request-model"))
```
### Option Precedence

When an option can be specified at multiple levels (client-level, request-level, or in request structures), the following precedence applies (highest to lowest):

1. **Request Structure Fields** (if applicable): Values set directly in request structures (e.g., `ChatRequest.Model`)
2. **Request-Level Options**: Options passed to individual method calls (e.g., `client.Chat(req, hfopts.WithModel("..."))`)
3. **Client-Level Options**: Options set when creating the Client (e.g., `hfgo.NewClient(hfopts.WithModel("..."))`)

This precedence ensures that more specific (request-level) configurations always override more general (client-level) configurations.

**Example**:
```go
// Client-level Model: "default-model"
client := hfgo.NewClient(hfopts.WithModel("default-model"))

// Request-level override: "request-model"
response, err := client.Chat(
    hftypes.ChatRequest{Messages: msgs},
    hfopts.WithModel("request-model"),
)
// Result: Uses "request-model"

// Request structure field: "structure-model"
response, err := client.Chat(
    hftypes.ChatRequest{
        Model: ptr("structure-model"),
        Messages: msgs,
    },
    hfopts.WithModel("request-model"),
)
// Result: Uses "structure-model" (highest precedence)
```

### Core Options
- `hfopts.WithBaseURL(url string)`: Base URL for API requests (no query params/fragments)
- `hfopts.WithToken(token string)`: Bearer authentication token
- `hfopts.WithModel(model string)`: Model identifier for requests
- `hfopts.WithProvider(provider hfproviders.Provider)`: Inference provider
  - On OpenAI-compatible endpoints (e.g. chat completions), a provider or selection policy can be pinned by appending a suffix to the model string (e.g. `model:sambanova`, `model:fastest`, `model:cheapest`, `model:preferred`); otherwise the HF router selects the provider
  - See https://huggingface.co/docs/inference-providers/main/en/index
- `hfopts.WithDefaultProvider()`: Sets the default HuggingFace provider (empty routing suffix)

### HTTP & Transport
- `hfopts.WithHTTPClientFactory(factory func() http.Client)`: Factory for HTTP clients
  - Invoked when options are applied
  - Should return fresh client value
  - Avoid sharing mutable internals like Transport unless synchronized
  - Nil factory results in nil HTTP client
- `hfopts.WithDefaultHTTPClient()`: Sets default HTTP client
- `hfopts.WithUserAgentSuffix(suffix string)`: Appends suffix to SDK user agent

### Context & Timeouts
- `hfopts.WithContext(ctx context.Context)`: Context for cancellation and timeouts
  - Nil context falls back to context.Background()

### Response Handling
- `hfopts.WithMaxResponseBodyBytes(n int64)`: Max bytes read from response body
  - Values <= 0 fall back to default

### Headers
- `hfopts.WithHeaders(h http.Header)`: Custom headers applied to all requests
  - Overrides existing values for matching keys
  - Per-request headers can still override
- `hfopts.WithHeader(key, value string)`: Single header applied to all requests
- `hfopts.WithDefaultHeader(key, value string)`: Header only if missing or empty

## Core Types

### ChatRequest
Represents a chat completion request. Key fields:

- `Model *string`: Model identifier (required)
- `Messages []ChatMessage`: Conversation history (required)
- `MaxTokens *int`: Max tokens in response (default: 1024, min: 0)
- `Temperature *float64`: Sampling temperature (0-2)
- `TopP *float64`: Nucleus sampling probability mass
- `TopLogProbs *int`: Number of most likely tokens (0-5, requires LogProbs=true)
- `FrequencyPenalty *float64`: Penalty for repeated tokens (-2.0 to 2.0)
- `PresencePenalty *float64`: Penalty for new topics (-2.0 to 2.0)
- `Stop []string`: Stop sequences (up to 4)
- `Seed *int64`: Deterministic sampling seed
- `Stream *bool`: Enable streaming (use ChatStream, not Chat)
- `StreamOptions *ChatStreamOptions`: SSE stream configuration
- `Tools []ChatTool`: Available tools/functions
- `ToolChoice *ChatToolChoice`: Tool selection behavior (auto, none, required, or function spec)
- `ToolPrompt *string`: Prompt appended before tools
- `LogProbs *bool`: Return log probabilities
- `ResponseFormat *ChatResponseFormat`: Response format (text, json_schema, json_object, or provider-specific)

**Validation**:
- Enforced in `MarshalJSON()` method
- Invalid payloads surface as configuration errors
- Model can be set via request field or client option (request field takes precedence)

### ChatResponse
Response from non-streaming chat completion. Fields:

- `ID string`: Response identifier
- `Model string`: Model used
- `Choices []ChatChoice`: Generated choices
- `Usage`: Token usage statistics

Validation:
- Enforced in `UnmarshalJSON()` method
- Invalid response payloads surface as SDK validation errors

### ChatStream
Wraps streaming chat completion response from `ChatStream()`. Defined in
`internal/chatstream` together with `ToolCallAccumulator`.

**Methods**:
- `Recv(ctx context.Context) (ChatStreamResponse, error)`: Blocks until next chunk arrives
  - Returns `io.EOF` when stream ends
  - Merges tool call metadata across deltas
- `Close() error`: Releases underlying HTTP connection and decoder goroutine
  - Must be called to promptly release resources
  - Safe to call on nil stream

**Tool Call Metadata Merging**:
- Automatically caches and merges tool call ID, type, and function name across streaming deltas
- Ensures each delta includes complete tool call metadata
- Implemented by `internal/chatstream.ToolCallAccumulator`

### ChatMessage
Represents a message in conversation history.

- `Role string`: Message role (system, user, assistant)
- `Content ChatMessageContent`: Message content
- `ToolCalls []ChatToolCall`: Tool calls made by assistant (if any)

### ChatTool
Represents a function tool available to the model.

- `Type string`: Tool type (currently "function")
- `Function ChatFunctionDefinition`: Function definition

### ChatToolChoice
Controls tool selection behavior. Supports:
- String values: "auto", "none", "required"
- Object: `{"function":{"name":"..."}}`
- Provider-specific values

### ChatResponseFormat
Response format specification. Known types:
- "text": Plain text response
- "json_schema": JSON with schema validation
- "json_object": JSON object response
- Provider-specific values accepted

### ChatStreamOptions
Configuration for streaming responses.

- `IncludeUsage *bool`: Include token usage in stream
### Event

Represents an SSE event returned by `hfraw.Client` stream methods.

- `Data []byte`: Event data payload
- `Event string`: Event type identifier
- `ID string`: Event ID
- `Retry *time.Duration`: Retry duration hint (if provided)

## Client API

All inference endpoints are called directly on a `Client` value. Each method
takes a request DTO and returns a typed response or stream; per-request options
are passed as variadic `hfopts.Option` values. The behavior below describes what the
Client methods perform. The low-level `hfraw.Client` escape hatch is documented separately.

### Chat

#### Chat(req ChatRequest, opts ...hfopts.Option) (ChatResponse, error)
Non-streaming chat completion.

**Concurrency and request mutation**:
- The request is passed **by value**; the SDK never mutates the caller's payload.
- The value copy shares nested data (slices, maps, pointed-to values) with the caller, so the request and the data it references must be treated as **read-only while a call is in flight**.
- Sequential, fully-awaited reuse of one request is safe and requires no cloning.
- For concurrent invocation, pass a defensive copy per call: `go client.Chat(req.Clone(), ...)`, or build a fresh request per call.

**Model and Provider Precedence**:
The Model field is resolved with the following precedence (highest to lowest):
1. ChatRequest.Model field (if non-nil and non-empty)
2. Per-request options Model override
3. Client-level Model option

The Provider field is applied as a suffix to the model string for routing in OpenAI-compatible endpoints. If the Model is in the format "model:provider", the Provider option is ignored. When the provider is the default HuggingFace provider, no suffix is appended and the router selects a provider automatically.

Provider selection is otherwise delegated to the HF router. To select a provider or selection policy explicitly, append a suffix to the model string (e.g. `model:sambanova`, `model:fastest`, `model:cheapest`, `model:preferred`). See https://huggingface.co/docs/inference-providers/main/en/index for the default (no-suffix) behavior.

**Behavior**:
- Returns `SDKError` (kind: Configuration) if the request is missing a model or messages (zero-value request)
- Applies per-request options to override client defaults
- Rejects requests with Stream=true (use ChatStream instead)
- Normalizes model and provider fields
- Returns `ChatResponse` with all choices and usage stats
- Returns `SDKError` (kind: Configuration) for invalid requests

#### ChatStream(req ChatRequest, opts ...hfopts.Option) (*ChatStream, error)
Streaming chat completion using SSE.

**Concurrency and request mutation**:
- The request is passed **by value**; the SDK never mutates the caller's payload.
- The request is fully consumed before the stream is returned, so the same sequential-reuse rules as `Chat` apply.
- For concurrent invocation, pass a defensive copy per call: `go client.ChatStream(req.Clone(), ...)`, or build a fresh request per call.

**Behavior**:
- Returns `SDKError` (kind: Configuration) if the request is missing a model or messages (zero-value request)
- Applies per-request options
- Always sends the request with streaming enabled
- Normalizes model and provider fields
- Returns `*ChatStream` for consuming chunks
- Caller must call `Close()` on returned stream
- Returns `SDKError` (kind: Configuration) for invalid requests

### Text Classification

#### ClassifyText(req TextClassificationRequest, opts ...hfopts.Option) ([]TextClassification, error)
Single text classification.

**Behavior**:
- Applies per-request options
- Returns flat array of classifications for the single input
- Automatically unwraps the response to get the single input result

#### ClassifyTextBatch(req TextClassificationBatchRequest, opts ...hfopts.Option) ([][]TextClassification, error)
Batch text classification for multiple inputs.

**API Response Format Normalization**:
The SDK handles a quirk in the HuggingFace API where the response format differs based on whether the `TopK` parameter is explicitly set:
- **When TopK is explicitly set**: Returns `[[classifications for input1], [classifications for input2], ...]` (per-input format)
- **When TopK is unset (nil)**: Returns `[[all classifications together]]` (flat format)

This inconsistency is handled transparently by the `normalizeTextClassificationResponse()` helper function.

### Question Answering

#### AnswerQuestion(req QuestionAnsweringRequest, opts ...hfopts.Option) ([]QuestionAnswering, error)
Question answering over a context passage.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a ranked list of answers extracted from the context
- Each answer includes the extracted text, score, and character span (start/end)
- The request `inputs` field is a structured object with `question` and `context` (both required)

### Token Classification

#### ClassifyTokens(req TokenClassificationRequest, opts ...hfopts.Option) ([]TokenClassification, error)
Single input token classification (named entity recognition).

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a list of classified tokens/entities for the single input
- Each entity includes its label, score, word text, and character span (start/end)
- When aggregation_strategy is "none", the `Entity` field is populated; otherwise `EntityGroup` is populated

#### ClassifyTokensBatch(req TokenClassificationBatchRequest, opts ...hfopts.Option) ([][]TokenClassification, error)
Batch token classification for multiple inputs.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a list of entity lists, one per input, in input order
- Callers should check the length of the response list before indexing

### Zero-Shot Text Classification

#### ZeroShotClassifyText(req ZeroShotTextClassificationRequest, opts ...hfopts.Option) ([]ZeroShotTextClassification, error)
Single input zero-shot text classification.

**Behavior**:
- Validates that candidate labels are provided in parameters
- Returns error if candidate labels are missing or empty
- Applies per-request options
- Returns flat array of classifications for the single input, ordered by score (descending)

#### ZeroShotClassifyTextBatch(req ZeroShotTextClassificationBatchRequest, opts ...hfopts.Option) ([][]ZeroShotTextClassification, error)
Batch zero-shot text classification for multiple inputs.

**API Response Normalization**:
The HuggingFace API returns batched zero-shot results in a different format than single inputs. The SDK transparently normalizes responses via `normalizeZeroShotTextClassificationResponse()`.

### Fill Mask

#### FillMask(req FillMaskRequest, opts ...hfopts.Option) ([]FillMaskPrediction, error)
Single input mask filling.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns ranked mask filling predictions for the single input

#### FillMaskBatch(req FillMaskBatchRequest, opts ...hfopts.Option) ([][]FillMaskPrediction, error)
Batch mask filling for multiple inputs.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a list of prediction lists, one per input, in input order
- Callers should check the length of the response list before indexing

### Summarization

#### Summarize(req SummarizationRequest, opts ...hfopts.Option) ([]Summarization, error)
Single text summarization.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a flat list of `Summarization` outputs for the single input

#### SummarizeBatch(req SummarizationBatchRequest, opts ...hfopts.Option) ([]Summarization, error)
Batch text summarization for multiple inputs.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- The API returns a flat list of `Summarization` outputs (one per input, in order) rather than a nested list, consistent with how it returns a list even for a single input

### Translation

#### Translate(req TranslationRequest, opts ...hfopts.Option) ([]Translation, error)
Single text translation.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a flat list of `Translation` outputs for the single input

#### TranslateBatch(req TranslationBatchRequest, opts ...hfopts.Option) ([]Translation, error)
Batch text translation for multiple inputs.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- The API returns a flat list of `Translation` outputs (one per input, in order) rather than a nested list, consistent with how it returns a list even for a single input

### Table Question Answering

#### AnswerTableQuestion(req TableQuestionAnsweringRequest, opts ...hfopts.Option) (TableQuestionAnswer, error)
Question answering over tabular data.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a single `TableQuestionAnswer` containing the answer text, cell values, coordinates, and optional aggregator
- The request `inputs` field is a structured object with `question` (string) and `table` (map of column names to string cell values), both required

**API Response Note**:
The HuggingFace API returns a bare JSON object for table question answering, not an array — despite the upstream schema declaring an array response. This method returns a single `TableQuestionAnswer` to match the actual API behavior.

### Feature Extraction

#### FeatureExtract(req FeatureExtractionRequest, opts ...hfopts.Option) (FeatureExtraction, error)
Single input feature extraction (embeddings).

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a single embedding vector (`[]float64`) for the input text
- The API returns a flat JSON array for a single input (not wrapped in an outer array)

**Parameters**:
- `normalize` (bool): Whether to normalize embeddings to unit L2 norm
- `prompt_name` (string): Name of the prompt for sentence-transformers encoding
- `truncate` (bool): Whether to truncate input to model's max length
- `truncation_direction` ("left" | "right"): Direction to truncate from

#### FeatureExtractBatch(req FeatureExtractionBatchRequest, opts ...hfopts.Option) ([]FeatureExtraction, error)
Batch feature extraction for multiple inputs.

**Behavior**:
- Applies per-request options
- Validates that a model is configured
- Returns a list of embedding vectors (`[][]float64`), one per input, in input order
- Callers should check the length of the response list before indexing

### `hfraw.Client` (escape hatch)

Created via `hfraw.NewClient(...)`. For raw HTTP requests without type-safe JSON handling. This is the advanced escape hatch for endpoints the SDK does not model. Its broader method matrix (`Do`/`DoRaw`/`DoReader`/`DoRawReader` and `Stream`/`StreamReader`/`StreamRaw`/`StreamRawReader`) is grouped under `hfraw.Client` rather than cluttering the root `Client` surface. `hfraw.Client`, `hfraw.Stream`, and `hfraw.Event` are defined in the `hfraw` package.

#### Do(requestBody []byte, method, path string, opts ...hfopts.Option) (*http.Response, error)
Raw request with error interpretation on non-2xx responses.

#### DoRaw(requestBody []byte, method, path string, opts ...hfopts.Option) (*http.Response, error)
Raw request without error interpretation (allows non-2xx responses).

#### DoReader(requestBody io.Reader, method, path string, opts ...hfopts.Option) (*http.Response, error)
Same as `Do`, but streams the request body from an `io.Reader`.

#### DoRawReader(requestBody io.Reader, method, path string, opts ...hfopts.Option) (*http.Response, error)
Same as `DoRaw`, but streams the request body from an `io.Reader`.

#### Stream(requestBody []byte, method, path string, opts ...hfopts.Option) (*hfraw.Stream, error)
SSE stream with error interpretation.

#### StreamReader(requestBody io.Reader, method, path string, opts ...hfopts.Option) (*hfraw.Stream, error)
Same as `Stream`, but streams the request body from an `io.Reader`.

#### StreamRaw(requestBody []byte, method, path string, opts ...hfopts.Option) (*hfraw.Stream, error)
SSE stream without error interpretation (allows non-2xx responses).

#### StreamRawReader(requestBody io.Reader, method, path string, opts ...hfopts.Option) (*hfraw.Stream, error)
Same as `StreamRaw`, but streams the request body from an `io.Reader`.

## Endpoints

Endpoints are resolved by the configured provider (see
[Provider Abstraction](#provider-abstraction)). The built-in
`HuggingFaceProvider` resolves them as follows, relative to the base URL
(default `https://router.huggingface.co`):

### Model endpoints
- **Path**: `hf-inference/models/{model}`
- **Tasks**: text classification, zero-shot text classification, token
  classification, question answering, table question answering, fill mask,
  summarization, translation
- **Method**: POST

### Pipeline endpoints
- **Path**: `hf-inference/models/{model}/pipeline/{task}`
- **Tasks**: feature extraction, sentence similarity
- **Method**: POST

### Chat Completions
- **Path**: `v1/chat/completions`
- **Method**: POST
- **Methods**: `Client.Chat(...)` or `Client.ChatStream(...)`

### Raw (escape hatch)

- `hfraw.NewClient(...)` accepts the same options as `hfgo.NewClient` and
  exposes arbitrary relative `path` endpoints not modeled by the SDK; the path
  is joined to the base URL configured via `hfopts.WithBaseURL`.

## Quality Assurance

### Testing Strategy

1. **Unit Tests**: Run with `go test ./...`
2. **Race Condition Detection**: Run with `go test -race ./...`
3. **Integration Tests**: Run with `-tags=integration`

### Linting & Code Quality
- `golangci-lint`: Comprehensive linting with custom config
- Test files excluded from specific linters (bodyclose, cyclop, errcheck, etc.)
- Examples excluded from revive, mnd, exhaustruct, errcheck, godoclint

## Development Commands

```bash
# Format
gofmt -s -w .

# Tidy
go mod tidy

# Vet
go vet ./...

# Lint
golangci-lint run --fix ./...

# Build
go build ./...

# Unit tests
go test -timeout 600s -v ./...

# Race tests
go test -race -timeout 600s -v ./...

# Integration tests (requires HF_TOKEN)
go test -tags=integration -timeout 600s -v ./...

# Coverage
go test -coverprofile=coverage.out -covermode=atomic ./...
```

## Best Practices for Development

### 1. Concurrency & Immutability
- Create new Client for different configurations, don't mutate
- Client methods snapshot the client's options on each call, so nothing needs to be cached or pre-bound

### 2. Error Handling
- Always type-assert errors to APIError or SDKError
- Use helper methods on APIError
- Always close Body on APIError

### 3. Concurrency & Request Safety

Request DTOs (e.g. `ChatRequest`) are passed to Client methods **by value**, and the SDK only ever mutates its own internal copy. This is the entire guarantee.

**What the SDK guarantees**:
- The SDK never mutates the request payload you pass in.
- A single immutable Client is safe for concurrent use.

**What the SDK does *not* guarantee, and why**:
- Because the value copy shares the request's nested data by reference (slices, maps, and pointed-to values), the SDK cannot prevent a caller from racing against an in-flight call by mutating that nested data.
- Go offers libraries no way to intercept ordinary field reads/writes on a plain struct, and retrofitting synchronization (mutexes, atomics, getters/setters) would make the DTOs non-copyable — destroying the value-copy semantics the whole SDK is built on. Preventing caller-authored races is therefore **structurally impossible** from inside the SDK; it is a caller responsibility.

**Safe usage patterns**:
1. **Read-only request**: Treat the request and the data it references as read-only while a call is in flight. Sequential, fully-awaited reuse is safe and needs no cloning.
2. **Defensive copy for concurrency**: Invoke with a per-call deep copy so each goroutine owns its request:
   ```go
   go client.Chat(req.Clone(), ...)      // each goroutine passes its own copy
   go client.ChatStream(req.Clone(), ...)
   ```
   `Clone()` is a **deep** copy: every slice gets new backing storage and every pointer a new pointee, so a clone shares nothing with its source. Exception: the value of an entry in `SummarizationParameters.GenerateParameters` or `TranslationParameters.GenerateParameters` (`map[string]any`) is shared, because its type is not known statically.
3. **No reuse**: Simply build a fresh request per call or per goroutine and never share request objects.

**Never**: share one mutable request object across goroutines and mutate it while calls are in flight — that is a data race the SDK cannot observe or prevent.

### 4. Streaming
- Always call Close() on `hftypes.ChatStream` or `hfraw.Stream`
- Prefer `defer stream.Close()` to ensure cleanup

### 5. Value Receivers vs Pointer Receivers
- Use value receivers for immutable types (Client, ChatRequest, etc.)
- Use pointer receivers for mutable types (`hftypes.ChatStream`, `hfraw.Stream`, etc.)

### 6. Generics
- Leverage Go generics for type-safe request/response handling

