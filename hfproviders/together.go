package hfproviders

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// togetherProviderName is the Together provider slug: the model-suffix token
// the chat task pins routing with, the Hub identifier the model resolver
// looks mappings up by, and the router path prefix the transcription endpoint
// is proxied under.
const togetherProviderName = "together"

// TogetherProvider serves chat completion and speech-recognition tasks
// through the Hugging Face router.
//
// Chat (unary and streaming, including vision-language requests) reuses
// [HuggingFaceChatMixin]'s endpoint and pins routing by appending the
// router's ":together" suffix to the model ID, exactly as the other chat
// partner providers do: the router receives the Hugging Face model ID plus
// suffix and performs provider mapping server-side. The suffix is a
// deterministic pin — the router's automatic failover applies to unsuffixed
// models and routing policies (":fastest", ":cheapest", ":preferred"), not to
// an explicit provider — so a model Together does not serve fails rather
// than running on another provider. Pass a request model with its own suffix
// (e.g. "model:preferred") to opt out of the pin.
//
// Chat responses are decoded like the default provider's, but requests go
// through a Together-specific codec: structured output for Together rides a
// flattened response_format shape ({"type":"json_schema","schema":...})
// rather than OpenAI's nested {"type":"json_schema",
// "json_schema":{"schema":...}} envelope — the official Hugging Face clients
// send the flattened shape, and this codec matches them. The router
// forwards the body untouched, so the rewrite happens here.
//
// Speech recognition is served through Together's OpenAI-compatible
// transcription endpoint, proxied by the router under the "together" path
// prefix, with Together's own model identifiers in the request body. Callers
// configure the Hugging Face model ID they know; the mixin translates it to
// Together's identifier at endpoint-construction time via its
// [ModelResolver], and the codec reuses the resolution from
// [EncodeParams.Mapping].
//
// Together also serves text-to-speech on the Hugging Face router, and the
// official clients carry task classes for embeddings and image/video
// generation, but this SDK does not implement text-to-speech and those
// tasks currently have no live Hub mappings for Together, so the provider
// does not expose them.
//
// Embedding the mixins keeps the promoted field TogetherProvider.Resolver
// unambiguous — it refers to the speech-recognition mixin's resolver — but
// prefer the constructor to wire it.
//
// The zero value is safe to use: chat works out of the box, and the
// pipeline endpoint method returns a configuration [hferrors.SDKError]
// instead of dereferencing a nil resolver. As with the Hugging Face mixins,
// the speech-recognition mixin can be embedded on its own in a custom
// provider that also implements Provider.Name and Provider.ProviderSuffix.
type TogetherProvider struct {
	HuggingFaceChatMixin
	TogetherSpeechRecognitionMixin
}

// Compile-time assertions that TogetherProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider              = (*TogetherProvider)(nil)
	_ ChatStreamProvider        = (*TogetherProvider)(nil)
	_ SpeechRecognitionProvider = (*TogetherProvider)(nil)
)

// NewTogetherProvider creates a [TogetherProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value TogetherProvider still serves chat; its
// speech-recognition endpoint method returns a configuration
// [hferrors.SDKError].
func NewTogetherProvider(resolver ModelResolver) (TogetherProvider, error) {
	if resolver == nil {
		return TogetherProvider{}, errNilResolver(togetherProviderName)
	}

	return TogetherProvider{
		HuggingFaceChatMixin{},
		TogetherSpeechRecognitionMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the Together provider for debug purposes.
func (p TogetherProvider) Name() string {
	return togetherProviderName
}

// ProviderSuffix returns the "together" provider suffix for appending to the
// model selection in chat completion tasks. Speech recognition routes via
// its endpoint path and does not use the suffix.
func (p TogetherProvider) ProviderSuffix() string {
	return togetherProviderName
}

// ChatCodec returns the Together chat codec: it encodes requests with
// Together's flattened response_format shape and decodes responses exactly
// like [HuggingFaceChatMixin]'s JSON codec. These methods shadow the embedded
// mixin's accessors, so the provider advertises the rewritten wire format
// while keeping the mixin's model-independent endpoint.
func (p TogetherProvider) ChatCodec() ChatCodec {
	return togetherChatCodec{}
}

// ChatStreamCodec returns the Together streaming chat codec, the streaming
// sibling of [TogetherProvider.ChatCodec].
func (p TogetherProvider) ChatStreamCodec() ChatStreamCodec {
	return togetherChatStreamCodec{}
}

// TogetherSpeechRecognitionMixin provides the Together endpoint and codec for
// speech recognition. It is a mixin: embed it in a type that also implements
// Provider.Name and Provider.ProviderSuffix to serve only speech
// recognition. Its [ModelResolver] maps the configured Hugging Face model ID
// to Together's provider model identifier at endpoint-construction time.
type TogetherSpeechRecognitionMixin struct {
	// Resolver maps the configured Hugging Face model ID to Together's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// SpeechRecognitionEndpoint resolves the configured Hugging Face model ID to
// Together's identifier and returns the router-proxied path for the
// transcription endpoint.
func (m TogetherSpeechRecognitionMixin) SpeechRecognitionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	mapping, err := togetherMapping(
		params.Context,
		m.Resolver,
		nil,
		params.Model,
		TaskAutomaticSpeechRecognition,
	)
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    "together/v1/audio/transcriptions",
		Mapping: mapping,
	}, nil
}

// SpeechRecognitionCodec returns the speech-recognition codec, which encodes
// requests as multipart/form-data and decodes JSON responses, reusing the
// mapping resolved at endpoint-construction time and falling back to the
// mixin's resolver when used standalone.
func (m TogetherSpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	return togetherSpeechRecognitionCodec{resolver: m.Resolver}
}

// togetherMapping resolves the configured Hugging Face model ID to Together's
// provider mapping. supplied — the mapping resolved while constructing the
// endpoint (see [Endpoint.Mapping] and [EncodeParams.Mapping]) — wins;
// otherwise the given resolver runs the lookup, mirroring DeepInfra's
// standalone-codec fallback.
func togetherMapping(
	ctx context.Context,
	resolver ModelResolver,
	supplied *ModelMapping,
	model string,
	task Task,
) (*ModelMapping, error) {
	if supplied != nil {
		return supplied, nil
	}
	if resolver == nil {
		return nil, errNilResolver(togetherProviderName)
	}

	mapping, err := resolver.ResolveModel(ctx, ResolveModelParams{
		Provider: togetherProviderName,
		Task:     task,
		Model:    model,
	})
	if err != nil {
		return nil, err
	}

	return &mapping, nil
}

// togetherResponseFormat is Together's flattened response_format wire shape:
// the schema rides directly under "schema" instead of OpenAI's nested
// "json_schema" envelope. Name and strict are dropped, matching the official
// clients' rewrite.
type togetherResponseFormat struct {
	Type   hftypes.ResponseFormatType `json:"type"`
	Schema json.RawMessage            `json:"schema"`
}

// togetherFlattenResponseFormat rewrites an already-marshaled chat request
// body so a json_schema response_format carries Together's flattened shape.
// Requests without a json_schema response_format (or whose envelope has no
// schema to unwrap — the official clients keep the envelope in that case) are
// returned unchanged. The rewrite round-trips the body through a raw-message
// map, which re-sorts top-level keys; key order carries no meaning here.
func togetherFlattenResponseFormat(
	format *hftypes.ChatResponseFormat,
	body []byte,
) ([]byte, error) {
	if format == nil ||
		format.Type != hftypes.ResponseFormatTypeJSONSchema ||
		format.JSONSchema == nil ||
		len(format.JSONSchema.Schema) == 0 {
		return body, nil
	}

	flat, err := json.Marshal(togetherResponseFormat{
		Type:   hftypes.ResponseFormatTypeJSONSchema,
		Schema: format.JSONSchema.Schema,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode request body",
			Err:     err,
		}
	}
	fields["response_format"] = flat

	rewritten, err := json.Marshal(fields)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return rewritten, nil
}

// Compile-time assertions that the Together chat codecs implement their task
// codecs.
var (
	_ ChatCodec       = (*togetherChatCodec)(nil)
	_ ChatStreamCodec = (*togetherChatStreamCodec)(nil)
)

// togetherChatCodec is a [Codec] for unary chat completion that encodes
// requests with Together's flattened json_schema response_format and decodes
// responses exactly like the default provider's JSON codec. Together omits
// system_fingerprint from some responses; [hftypes.ChatResponse] treats it as
// optional, so decoding needs no adaptation.
type togetherChatCodec struct{}

func (c togetherChatCodec) Encode(
	params EncodeParams[hftypes.ChatRequest],
) (body []byte, headers http.Header, err error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}

	body, headers, err = inner.Encode(params)
	if err != nil {
		return nil, nil, err
	}

	body, err = togetherFlattenResponseFormat(params.Request.ResponseFormat, body)
	if err != nil {
		return nil, nil, err
	}

	return body, headers, nil
}

func (c togetherChatCodec) Decode(params DecodeParams) (hftypes.ChatResponse, error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}

	return inner.Decode(params)
}

// togetherChatStreamCodec is the streaming sibling of [togetherChatCodec]:
// it encodes requests identically and decodes SSE payloads exactly like the
// default provider's JSON codec.
type togetherChatStreamCodec struct{}

func (c togetherChatStreamCodec) Encode(
	params EncodeParams[hftypes.ChatRequest],
) (body []byte, headers http.Header, err error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}

	body, headers, err = inner.Encode(params)
	if err != nil {
		return nil, nil, err
	}

	body, err = togetherFlattenResponseFormat(params.Request.ResponseFormat, body)
	if err != nil {
		return nil, nil, err
	}

	return body, headers, nil
}

func (c togetherChatStreamCodec) Decode(params DecodeParams) (hftypes.ChatStreamResponse, error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}

	return inner.Decode(params)
}

// Compile-time assertion that the Together speech codec implements its task
// codec.
var _ SpeechRecognitionCodec = (*togetherSpeechRecognitionCodec)(nil)

// togetherSpeechRecognitionCodec is a [Codec] for speech recognition that
// encodes canonical requests as multipart/form-data for Together's
// OpenAI-compatible transcription endpoint and decodes its JSON responses.
// The wire format is the shared one implemented by the openAI* transcription
// helpers: the body carries the decoded audio on the "file" part with a
// media-type-derived filename, the recognition parameters as form fields, and
// the mapped provider model identifier on the "model" field — applied last,
// so provider parameters cannot override it.
type togetherSpeechRecognitionCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c togetherSpeechRecognitionCodec) Encode(
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	mapping, err := togetherMapping(
		params.Context,
		c.resolver,
		params.Mapping,
		params.Model,
		TaskAutomaticSpeechRecognition,
	)
	if err != nil {
		return nil, nil, err
	}

	return openAIEncodeTranscription(params.Request, mapping)
}

func (c togetherSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return openAIDecodeTranscription(params)
}
