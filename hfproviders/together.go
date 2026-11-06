package hfproviders

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// togetherProviderName is the Together provider slug and the model-suffix
// token appended to chat model IDs, matching the router's provider
// identifiers.
const togetherProviderName = "together"

// TogetherProvider serves chat completion tasks through the Hugging Face
// router by implementing [ChatProvider] and [ChatStreamProvider]. It reuses
// [HuggingFaceChatMixin]'s endpoint because the OpenAI-compatible schema
// travels in the request body: the router performs provider mapping
// server-side, so no [ModelResolver] is required. Routing is pinned to
// Together by appending the ":together" provider suffix to the model ID.
// The suffix is a deterministic pin — the router's automatic failover applies
// to unsuffixed models and routing policies (":fastest", ":cheapest",
// ":preferred"), not to an explicit provider — so a model Together does not
// serve fails rather than running on another provider. Pass a request model
// with its own suffix (e.g. "model:preferred") to opt out of the pin.
//
// Unlike the other chat partners, Together needs codecs of its own:
// structured output rides a flattened response_format shape
// ({"type":"json_schema","schema":...}) rather than OpenAI's nested
// {"type":"json_schema","json_schema":{"schema":...}} envelope — the official
// Hugging Face clients send the flattened shape, and this provider's codecs
// match them. The router forwards bodies untouched, so the rewrite happens
// client-side. Responses decode like the default provider's, including
// vision-language content.
//
// The zero value is ready to use, as with the other chat partner providers.
type TogetherProvider struct {
	HuggingFaceChatMixin
}

// Compile-time assertions that TogetherProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider       = (*TogetherProvider)(nil)
	_ ChatStreamProvider = (*TogetherProvider)(nil)
)

// NewTogetherProvider returns a TogetherProvider that serves chat completion
// (unary and streaming) pinned to Together through the ":together" model
// suffix.
func NewTogetherProvider() TogetherProvider {
	return TogetherProvider{}
}

// Name returns the name of the Together provider for debug purposes.
func (p TogetherProvider) Name() string {
	return togetherProviderName
}

// ProviderSuffix returns the "together" provider suffix for appending to the
// model selection in chat completion tasks.
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

// togetherChatCodec is a [Codec] for unary chat completion that merges the
// request's ProviderParams into the body's top level (with the resolved
// "model" key protected, like [HuggingFaceChatMixin]) and encodes requests
// with Together's flattened json_schema response_format; responses decode
// exactly like the default provider's JSON codec. Together omits
// system_fingerprint from some responses; [hftypes.ChatResponse] treats it as
// optional, so decoding needs no adaptation.
type togetherChatCodec struct{}

func (c togetherChatCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.ChatRequest],
) (body []byte, headers http.Header, err error) {
	inner := hfChatParamsCodec(
		JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
		chatRequestProviderParams,
	)

	body, headers, err = inner.Encode(ctx, params)
	if err != nil {
		return nil, nil, err
	}

	body, err = togetherFlattenResponseFormat(params.Request.ResponseFormat, body)
	if err != nil {
		return nil, nil, err
	}

	return body, headers, nil
}

func (c togetherChatCodec) Decode(
	ctx context.Context,
	params DecodeParams,
) (hftypes.ChatResponse, error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}

	return inner.Decode(ctx, params)
}

// togetherChatStreamCodec is the streaming sibling of [togetherChatCodec]:
// it encodes requests identically (ProviderParams merge, flattened
// response_format) and decodes SSE payloads exactly like the default
// provider's JSON codec.
type togetherChatStreamCodec struct{}

func (c togetherChatStreamCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.ChatRequest],
) (body []byte, headers http.Header, err error) {
	inner := hfChatParamsCodec(
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		chatRequestProviderParams,
	)

	body, headers, err = inner.Encode(ctx, params)
	if err != nil {
		return nil, nil, err
	}

	body, err = togetherFlattenResponseFormat(params.Request.ResponseFormat, body)
	if err != nil {
		return nil, nil, err
	}

	return body, headers, nil
}

func (c togetherChatStreamCodec) Decode(
	ctx context.Context,
	params DecodeParams,
) (hftypes.ChatStreamResponse, error) {
	inner := JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{}

	return inner.Decode(ctx, params)
}
