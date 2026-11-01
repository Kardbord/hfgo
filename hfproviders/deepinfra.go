package hfproviders

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"slices"
	"strconv"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// deepinfraProviderName is the DeepInfra provider slug: the model-suffix
// token the chat task pins routing with, and the Hub identifier the model
// resolver looks mappings up by.
const deepinfraProviderName = "deepinfra"

// deepInfraFLACSignature is the magic prefix of a FLAC stream. net/http's
// byte sniffer has no FLAC signature, so the speech-recognition codec checks
// it explicitly to label FLAC input correctly.
const deepInfraFLACSignature = "fLaC"

// deepInfraDefaultAudioExtension is the filename extension used for audio
// whose media type is unknown, matching the official clients' fallback.
const deepInfraDefaultAudioExtension = "wav"

// DeepInfraProvider serves chat completion and speech-recognition, and pooled
// feature-extraction tasks through the Hugging Face router.
//
// Chat (unary and streaming, including vision-language requests) reuses
// [HuggingFaceChatMixin] and pins routing by appending the router's
// ":deepinfra" suffix to the model ID, exactly as the other chat partner
// providers do: the router receives the Hugging Face model ID plus suffix and
// performs provider mapping server-side. The suffix is a deterministic pin —
// the router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model DeepInfra does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin. This routes through the unified
// OpenAI-compatible chat endpoint rather than the provider-scoped path the
// official Hugging Face clients use for DeepInfra, matching the convention
// shared by every other chat partner provider in this package.
//
// Speech recognition and feature extraction are served through DeepInfra's
// OpenAI-compatible endpoints, proxied by the router under the "deepinfra"
// path prefix, with DeepInfra's own model identifiers in the request body.
// Callers configure the Hugging Face model ID they know; each pipeline mixin
// translates it to DeepInfra's identifier at endpoint-construction time via
// its [ModelResolver], and the codecs reuse the resolution from
// [EncodeParams.Mapping]. The two codecs deviate from the official clients in
// one deliberate way: the hf-inference feature-extraction parameters the
// canonical request carries (e.g. normalize) are not forwarded, as
// DeepInfra's embeddings API does not accept them — DeepInfra-specific
// parameters such as encoding_format ride the request's ProviderParams field
// instead. On both pipeline tasks the mapped model is applied to the payload
// last, so ProviderParams entries cannot override it.
//
// Embedding all three mixins means the promoted field DeepInfraProvider
// would derive for Resolver is ambiguous — access each mixin's resolver
// explicitly if you ever need to (e.g. p.DeepInfraSpeechRecognitionMixin.Resolver).
//
// The zero value is safe to use: pipeline endpoint methods return a
// configuration [hferrors.SDKError] instead of dereferencing a nil resolver.
// As with the Hugging Face mixins, each DeepInfra mixin can be embedded on
// its own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
//
// DeepInfra also serves text-to-speech on the Hugging Face router; this SDK
// does not implement that task yet, so the provider does not expose it.
type DeepInfraProvider struct {
	HuggingFaceChatMixin
	DeepInfraSpeechRecognitionMixin
	DeepInfraFeatureExtractionMixin
}

// Compile-time assertions that DeepInfraProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider                         = (*DeepInfraProvider)(nil)
	_ ChatStreamProvider                   = (*DeepInfraProvider)(nil)
	_ SpeechRecognitionProvider            = (*DeepInfraProvider)(nil)
	_ FeatureExtractionPooledProvider      = (*DeepInfraProvider)(nil)
	_ FeatureExtractionPooledBatchProvider = (*DeepInfraProvider)(nil)
)

// NewDeepInfraProvider creates a [DeepInfraProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value DeepInfraProvider is still safe to call; its
// pipeline endpoint methods return a configuration [hferrors.SDKError].
func NewDeepInfraProvider(resolver ModelResolver) (DeepInfraProvider, error) {
	if resolver == nil {
		return DeepInfraProvider{}, errNilResolver(deepinfraProviderName)
	}

	return DeepInfraProvider{
		HuggingFaceChatMixin:            HuggingFaceChatMixin{},
		DeepInfraSpeechRecognitionMixin: DeepInfraSpeechRecognitionMixin{Resolver: resolver},
		DeepInfraFeatureExtractionMixin: DeepInfraFeatureExtractionMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the DeepInfra provider for debug purposes.
func (p DeepInfraProvider) Name() string {
	return deepinfraProviderName
}

// ProviderSuffix returns the "deepinfra" provider suffix for appending to the
// model selection in chat completion tasks. Pipeline tasks route via their
// endpoint paths and do not use the suffix.
func (p DeepInfraProvider) ProviderSuffix() string {
	return deepinfraProviderName
}

// DeepInfraSpeechRecognitionMixin provides the DeepInfra endpoint and codec
// for speech recognition. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only speech
// recognition. Its [ModelResolver] maps the configured Hugging Face model ID
// to DeepInfra's provider model identifier at endpoint-construction time.
type DeepInfraSpeechRecognitionMixin struct {
	// Resolver maps the configured Hugging Face model ID to DeepInfra's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// SpeechRecognitionEndpoint resolves the configured Hugging Face model ID to
// DeepInfra's identifier and returns the router-proxied path for the
// transcription endpoint.
func (m DeepInfraSpeechRecognitionMixin) SpeechRecognitionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveDeepInfraEndpoint(
		m.Resolver,
		params,
		TaskAutomaticSpeechRecognition,
		"deepinfra/v1/openai/audio/transcriptions",
	)
}

// SpeechRecognitionCodec returns the speech-recognition codec, which encodes
// requests as multipart/form-data and decodes JSON responses, reusing the
// mapping resolved at endpoint-construction time and falling back to the
// mixin's resolver when used standalone.
func (m DeepInfraSpeechRecognitionMixin) SpeechRecognitionCodec() SpeechRecognitionCodec {
	return deepinfraSpeechRecognitionCodec{resolver: m.Resolver}
}

// DeepInfraFeatureExtractionMixin provides the DeepInfra endpoint and codecs
// for pooled feature extraction, including its batch variant. It is a mixin:
// embed it in a type that also implements Provider.Name and
// Provider.ProviderSuffix to serve only feature extraction. Its
// [ModelResolver] maps the configured Hugging Face model ID to DeepInfra's
// provider model identifier at endpoint-construction time.
//
// Only pooled output is implemented: DeepInfra's OpenAI-compatible embeddings
// endpoint returns one vector per input, never per-token embeddings, so this
// mixin does not satisfy the token-level feature-extraction provider
// interfaces.
type DeepInfraFeatureExtractionMixin struct {
	// Resolver maps the configured Hugging Face model ID to DeepInfra's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint methods are called.
	Resolver ModelResolver
}

// FeatureExtractionEndpoint resolves the configured Hugging Face model ID to
// DeepInfra's identifier and returns the router-proxied path for the
// embeddings endpoint.
func (m DeepInfraFeatureExtractionMixin) FeatureExtractionEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveDeepInfraEndpoint(
		m.Resolver,
		params,
		TaskFeatureExtraction,
		"deepinfra/v1/openai/embeddings",
	)
}

// FeatureExtractionBatchEndpoint resolves the configured Hugging Face model
// ID to DeepInfra's identifier and returns the router-proxied path for the
// embeddings endpoint. Single and batch requests share the same path.
func (m DeepInfraFeatureExtractionMixin) FeatureExtractionBatchEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveDeepInfraEndpoint(
		m.Resolver,
		params,
		TaskFeatureExtraction,
		"deepinfra/v1/openai/embeddings",
	)
}

// FeatureExtractionPooledCodec returns the pooled codec for the single-input
// FeatureExtraction task, which encodes requests as DeepInfra's OpenAI-style
// embeddings JSON body and decodes responses into a pooled embedding.
func (m DeepInfraFeatureExtractionMixin) FeatureExtractionPooledCodec() FeatureExtractionPooledCodec {
	return deepinfraFeatureExtractionCodec{resolver: m.Resolver}
}

// FeatureExtractionPooledBatchCodec returns the pooled codec for the
// FeatureExtractionBatch task, which encodes requests as DeepInfra's
// OpenAI-style embeddings JSON body and decodes responses into one pooled
// embedding per input.
func (m DeepInfraFeatureExtractionMixin) FeatureExtractionPooledBatchCodec() FeatureExtractionPooledBatchCodec {
	return deepinfraFeatureExtractionBatchCodec{resolver: m.Resolver}
}

// resolveDeepInfraEndpoint resolves the configured Hugging Face model ID to
// DeepInfra's provider model identifier and returns the router-proxied
// endpoint at the given fixed path. The provider slug given to the resolver
// is DeepInfra's Hub identifier — the key the Hub's mapping data uses — which
// is a provider fact, not something the SDK derives.
func resolveDeepInfraEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
	path string,
) (Endpoint, error) {
	mapping, err := deepInfraMapping(params.Context, resolver, nil, params.Model, task)
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    path,
		Mapping: mapping,
	}, nil
}

// deepInfraMapping resolves the configured Hugging Face model ID to
// DeepInfra's provider mapping. supplied — the mapping resolved while
// constructing the endpoint (see [Endpoint.Mapping] and
// [EncodeParams.Mapping]) — wins; otherwise the given resolver runs the
// lookup, mirroring WaveSpeed's standalone-codec fallback.
func deepInfraMapping(
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
		return nil, errNilResolver(deepinfraProviderName)
	}

	mapping, err := resolver.ResolveModel(ctx, ResolveModelParams{
		Provider: deepinfraProviderName,
		Task:     task,
		Model:    model,
	})
	if err != nil {
		return nil, err
	}

	return &mapping, nil
}

// Compile-time assertions that the DeepInfra codecs implement their task
// codecs.
var (
	_ SpeechRecognitionCodec            = (*deepinfraSpeechRecognitionCodec)(nil)
	_ FeatureExtractionPooledCodec      = (*deepinfraFeatureExtractionCodec)(nil)
	_ FeatureExtractionPooledBatchCodec = (*deepinfraFeatureExtractionBatchCodec)(nil)
)

// deepinfraSpeechRecognitionCodec is a [Codec] for speech recognition that
// encodes canonical requests as multipart/form-data for DeepInfra's
// OpenAI-compatible transcription endpoint and decodes its JSON responses.
//
// The body carries the decoded audio on the "file" part with a
// media-type-derived filename, the recognition parameters as form fields, and
// the mapped provider model identifier on the "model" field — applied last,
// so provider parameters cannot override it. This is the only wire format
// DeepInfra's transcriptions endpoint accepts, unlike the Hugging Face
// provider's raw-bytes or JSON variants.
type deepinfraSpeechRecognitionCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c deepinfraSpeechRecognitionCodec) Encode(
	params EncodeParams[hftypes.SpeechRecognitionRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	mapping, err := deepInfraMapping(
		params.Context,
		c.resolver,
		params.Mapping,
		params.Model,
		TaskAutomaticSpeechRecognition,
	)
	if err != nil {
		return nil, nil, err
	}

	audio, err := base64.StdEncoding.DecodeString(req.Input)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode base64 input",
			Err:     err,
		}
	}

	// Sniff the audio's media type from its magic bytes: it drives both the
	// file part's Content-Type and the part's filename extension. net/http's
	// sniffer has no FLAC signature, so a bare "fLaC" prefix is checked
	// explicitly; everything else falls through
	// [deepInfraAudioExtension]'s table and ".wav" fallback.
	sniffed := http.DetectContentType(audio)
	if sniffed == "application/octet-stream" &&
		bytes.HasPrefix(audio, []byte(deepInfraFLACSignature)) {
		sniffed = "audio/flac"
	}

	fields, err := deepInfraTranscriptionFields(&req)
	if err != nil {
		return nil, nil, err
	}
	// Applied last so a caller-supplied entry cannot override the mapped
	// provider model, matching the official clients.
	fields["model"] = mapping.ProviderModelID

	body, contentType, err := deepInfraMultipartBody(audio, sniffed, fields)
	if err != nil {
		return nil, nil, err
	}

	headers = http.Header{}
	headers.Set("Content-Type", contentType)
	headers.Set("Accept", mimeTypeJSON)

	return body, headers, nil
}

func (c deepinfraSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return deepInfraDecodeTranscription(params)
}

// deepInfraTranscriptionFields collects the recognition parameters and
// provider parameters as form-field values, skipping nil provider entries.
// The mapped model is not part of the result: the caller applies it last so
// provider parameters cannot override it.
func deepInfraTranscriptionFields(
	req *hftypes.SpeechRecognitionRequest,
) (map[string]string, error) {
	fields := map[string]string{}

	if req.Parameters != nil && req.Parameters.ReturnTimestamps != nil {
		fields["return_timestamps"] = strconv.FormatBool(*req.Parameters.ReturnTimestamps)
	}

	for key, value := range req.ProviderParams {
		if value == nil {
			continue
		}

		formatted, err := deepInfraFormFieldValue(value)
		if err != nil {
			return nil, err
		}

		fields[key] = formatted
	}

	return fields, nil
}

// deepInfraTranscriptionSegment is one segment of DeepInfra's transcription
// response.
type deepInfraTranscriptionSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// deepInfraDecodeTranscription decodes the transcription endpoint's JSON
// response into the canonical response: the transcript rides on "text" and
// segments map onto canonical chunks with [start, end] second timestamps.
// Every other top-level field (e.g. usage, duration, language) rides
// ProviderMetadata. A body without a string transcript is a serialization
// error.
func deepInfraDecodeTranscription(params DecodeParams) (hftypes.SpeechRecognitionResponse, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(params.Body, &fields); err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	text, err := deepInfraTranscriptionText(fields)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	var chunks []hftypes.SpeechRecognitionChunk
	if segmentsRaw, present := fields["segments"]; present && !isJSONNull(segmentsRaw) {
		chunks, err = deepInfraTranscriptionChunks(segmentsRaw)
		if err != nil {
			return hftypes.SpeechRecognitionResponse{}, err
		}
	}

	metadata, err := deepInfraTranscriptionMetadata(fields)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	resp := hftypes.SpeechRecognitionResponse{
		Text:             text,
		Chunks:           chunks,
		ProviderMetadata: nil,
	}
	if len(metadata) > 0 {
		resp.ProviderMetadata = metadata
	}

	return resp, nil
}

// deepInfraTranscriptionText extracts the response's transcript. A missing or
// non-string "text" field is a serialization error.
func deepInfraTranscriptionText(fields map[string]json.RawMessage) (string, error) {
	textRaw, present := fields["text"]
	if !present || isJSONNull(textRaw) {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "transcription response missing text",
			Err:     nil,
		}
	}

	var text string
	if err := json.Unmarshal(textRaw, &text); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "transcription response text must be a string",
			Err:     err,
		}
	}

	return text, nil
}

// deepInfraTranscriptionChunks maps the response's segments onto canonical
// chunks with [start, end] second timestamps. A non-array segments field is
// a serialization error; the caller leaves chunks unset when the field is
// absent or null.
func deepInfraTranscriptionChunks(
	segmentsRaw json.RawMessage,
) ([]hftypes.SpeechRecognitionChunk, error) {
	var segments []deepInfraTranscriptionSegment
	if err := json.Unmarshal(segmentsRaw, &segments); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	chunks := make([]hftypes.SpeechRecognitionChunk, 0, len(segments))
	for _, segment := range segments {
		chunks = append(chunks, hftypes.SpeechRecognitionChunk{
			Text:      segment.Text,
			Timestamp: []float64{segment.Start, segment.End},
		})
	}

	return chunks, nil
}

// deepInfraTranscriptionMetadata collects every top-level response field
// other than the canonical ones (text, segments). The returned map may be
// empty when the response carries nothing beyond the canonical fields; the
// caller only attaches it when there is something in it.
func deepInfraTranscriptionMetadata(
	fields map[string]json.RawMessage,
) (map[string]any, error) {
	metadata := make(map[string]any)
	for key, raw := range fields {
		if key == "text" || key == "segments" {
			continue
		}

		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: errFailedToDecodeResponseBody,
				Err:     err,
			}
		}

		metadata[key] = value
	}

	return metadata, nil
}

// deepInfraAudioExtension returns the filename extension for the given audio
// media type, defaulting to ".wav" for unknown or empty types. The mapping
// mirrors huggingface.js's media-type table, extended with the spellings
// net/http's byte sniffer actually emits: mp4 and webm containers are labeled
// video/*.
func deepInfraAudioExtension(mimeType string) string {
	mediaType, ok := parseMediaType(mimeType)
	if ok {
		switch mediaType {
		case "audio/mpeg", "audio/mp3":
			return "mp3"
		case "audio/mp4", "video/mp4":
			return "mp4"
		case "audio/m4a", "audio/x-m4a":
			return "m4a"
		case "audio/flac", "audio/x-flac":
			return "flac"
		case "audio/ogg", "application/ogg":
			return "ogg"
		case "audio/wav", "audio/wave", "audio/x-wav":
			return "wav"
		case "audio/webm", "video/webm":
			return "webm"
		}
	}

	return deepInfraDefaultAudioExtension
}

// deepInfraFormFieldValue formats one form-field value the way the official
// clients do: strings pass through unchanged, booleans and numbers stringify,
// and any other value JSON-encodes.
func deepInfraFormFieldValue(value any) (string, error) {
	switch typed := value.(type) {
	case string:
		return typed, nil
	case bool:
		return strconv.FormatBool(typed), nil
	case json.Number:
		return typed.String(), nil
	}

	if formatted, ok := deepInfraIntFormFieldValue(value); ok {
		return formatted, nil
	}
	if formatted, ok := deepInfraUintFormFieldValue(value); ok {
		return formatted, nil
	}
	if formatted, ok := deepInfraFloatFormFieldValue(value); ok {
		return formatted, nil
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to encode provider parameter as a form value",
			Err:     err,
		}
	}

	return string(encoded), nil
}

// deepInfraIntFormFieldValue stringifies signed integers of every width. ok
// is false for values that are not signed integers.
func deepInfraIntFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case int:
		return strconv.FormatInt(int64(typed), 10), true
	case int8:
		return strconv.FormatInt(int64(typed), 10), true
	case int16:
		return strconv.FormatInt(int64(typed), 10), true
	case int32:
		return strconv.FormatInt(int64(typed), 10), true
	case int64:
		return strconv.FormatInt(typed, 10), true
	}

	return "", false
}

// deepInfraUintFormFieldValue stringifies unsigned integers of every width.
// ok is false for values that are not unsigned integers.
func deepInfraUintFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case uint:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint8:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint16:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint32:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint64:
		return strconv.FormatUint(typed, 10), true
	}

	return "", false
}

// deepInfraFloatFormFieldValue stringifies floats of both widths. ok is false
// for values that are not floats.
func deepInfraFloatFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case float32:
		return strconv.FormatFloat(float64(typed), 'g', -1, 32), true
	case float64:
		return strconv.FormatFloat(typed, 'g', -1, 64), true
	}

	return "", false
}

// deepInfraMultipartBody builds the transcription endpoint's multipart body:
// the audio rides on the "file" part — its Content-Type set to the sniffed
// media type and its filename derived from it — followed by the remaining
// form fields in sorted key order, matching the official clients' layout.
func deepInfraMultipartBody(
	audio []byte,
	mimeType string,
	fields map[string]string,
) (body []byte, contentType string, err error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	partHeader := textproto.MIMEHeader{}
	partHeader.Set(
		"Content-Disposition",
		fmt.Sprintf(
			`form-data; name="file"; filename="audio.%s"`,
			deepInfraAudioExtension(mimeType),
		),
	)
	partHeader.Set("Content-Type", mimeType)
	part, err := writer.CreatePart(partHeader)
	if err != nil {
		return nil, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}
	if _, err := part.Write(audio); err != nil {
		return nil, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	for _, key := range slices.Sorted(maps.Keys(fields)) {
		if err := writer.WriteField(key, fields[key]); err != nil {
			return nil, "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: errFailedToMarshalRequestBody,
				Err:     err,
			}
		}
	}

	if err := writer.Close(); err != nil {
		return nil, "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}

// deepinfraFeatureExtractionCodec is a [Codec] for the pooled output of the
// single-input FeatureExtraction task. It encodes canonical requests as
// DeepInfra's OpenAI-style embeddings JSON body and decodes responses into a
// pooled embedding.
type deepinfraFeatureExtractionCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c deepinfraFeatureExtractionCodec) Encode(
	params EncodeParams[hftypes.FeatureExtractionRequest],
) (body []byte, headers http.Header, err error) {
	mapping, err := deepInfraMapping(
		params.Context,
		c.resolver,
		params.Mapping,
		params.Model,
		TaskFeatureExtraction,
	)
	if err != nil {
		return nil, nil, err
	}

	return deepInfraEncodeEmbeddings(params.Request.Input, params.Request.ProviderParams, mapping)
}

func (c deepinfraFeatureExtractionCodec) Decode(
	params DecodeParams,
) (hftypes.FeatureExtractionPooledEmbedding, error) {
	embeddings, err := deepInfraDecodeEmbeddings(params)
	if err != nil {
		return nil, err
	}

	return embeddings[0], nil
}

// deepinfraFeatureExtractionBatchCodec is a [Codec] for the pooled output of
// the FeatureExtractionBatch task. It encodes canonical requests as
// DeepInfra's OpenAI-style embeddings JSON body and decodes responses into
// one pooled embedding per input, in order.
type deepinfraFeatureExtractionBatchCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c deepinfraFeatureExtractionBatchCodec) Encode(
	params EncodeParams[hftypes.FeatureExtractionBatchRequest],
) (body []byte, headers http.Header, err error) {
	mapping, err := deepInfraMapping(
		params.Context,
		c.resolver,
		params.Mapping,
		params.Model,
		TaskFeatureExtraction,
	)
	if err != nil {
		return nil, nil, err
	}

	return deepInfraEncodeEmbeddings(params.Request.Inputs, params.Request.ProviderParams, mapping)
}

func (c deepinfraFeatureExtractionBatchCodec) Decode(
	params DecodeParams,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	return deepInfraDecodeEmbeddings(params)
}

// deepInfraEncodeEmbeddings builds the embeddings endpoint's OpenAI-style
// JSON body: "input" carries the text(s), provider parameters merge at the
// top level, and "model" is applied last from the resolved mapping so
// provider parameters cannot override it. The hf-inference feature-extraction
// parameters the canonical request types carry (e.g. normalize) are not
// forwarded — DeepInfra's embeddings API does not accept them;
// DeepInfra-specific parameters such as encoding_format ride ProviderParams
// instead. T mirrors the OpenAI input field's two shapes: a single text for
// the single-input task, a batch of texts for the batch task.
func deepInfraEncodeEmbeddings[T string | []string](
	input T,
	providerParams map[string]any,
	mapping *ModelMapping,
) (body []byte, headers http.Header, err error) {
	payload := map[string]any{}
	for key, value := range providerParams {
		if value == nil {
			continue
		}

		payload[key] = value
	}
	payload["input"] = input
	payload["model"] = mapping.ProviderModelID

	body, err = json.Marshal(payload)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	headers = http.Header{}
	headers.Set("Content-Type", mimeTypeJSON)
	headers.Set("Accept", mimeTypeJSON)

	return body, headers, nil
}

// deepInfraEmbeddingEntry is one embedding in the embeddings endpoint's
// response.
type deepInfraEmbeddingEntry struct {
	Embedding []float64 `json:"embedding"`
}

// deepInfraEmbeddingsResponse is the embeddings endpoint's response shape: a
// data array holding one embedding per input, in order.
type deepInfraEmbeddingsResponse struct {
	Data []deepInfraEmbeddingEntry `json:"data"`
}

// deepInfraDecodeEmbeddings decodes the embeddings endpoint's response into
// pooled embeddings, one per data entry. A body without embeddings — or an
// entry without an embedding array — is a serialization error.
func deepInfraDecodeEmbeddings(
	params DecodeParams,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
	}

	var resp deepInfraEmbeddingsResponse
	if err := json.Unmarshal(params.Body, &resp); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	if len(resp.Data) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "embeddings response contains no embeddings",
			Err:     nil,
		}
	}

	embeddings := make([]hftypes.FeatureExtractionPooledEmbedding, 0, len(resp.Data))
	for i, entry := range resp.Data {
		if entry.Embedding == nil {
			return nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: fmt.Sprintf("embeddings response entry %d is missing an embedding", i),
				Err:     nil,
			}
		}

		embeddings = append(embeddings, hftypes.FeatureExtractionPooledEmbedding(entry.Embedding))
	}

	return embeddings, nil
}
