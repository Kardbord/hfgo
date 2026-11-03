package hfproviders

import (
	"bytes"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hftypes"
)

// deepinfraProviderName is the DeepInfra provider slug: the model-suffix
// token the chat task pins routing with, and the Hub identifier the model
// resolver looks mappings up by.
const deepinfraProviderName = "deepinfra"

// DeepInfra's router-proxied task endpoints. The OpenAI-compatible paths
// embed the router's deepinfra prefix; single and batch embeddings share the
// same path.
const (
	deepinfraTranscriptionsPath = deepinfraProviderName + "/v1/openai/audio/transcriptions"
	deepinfraEmbeddingsPath     = deepinfraProviderName + "/v1/openai/embeddings"
)

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
// Embedding all three mixins means the promoted field
// DeepInfraProvider.Resolver is ambiguous — access each mixin's resolver
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
		deepinfraTranscriptionsPath,
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
		deepinfraEmbeddingsPath,
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
		deepinfraEmbeddingsPath,
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
	mapping, err := resolveMapping(
		params.Context,
		deepinfraProviderName,
		resolver,
		nil,
		task,
		params.Model,
	)
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    path,
		Mapping: mapping,
	}, nil
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
	mapping, err := resolveMapping(
		params.Context,
		deepinfraProviderName,
		c.resolver,
		params.Mapping,
		TaskAutomaticSpeechRecognition,
		params.Model,
	)
	if err != nil {
		return nil, nil, err
	}

	var bodyBuf bytes.Buffer
	headers, err = encodeOpenAITranscription(&bodyBuf, params.Request, mapping)
	if err != nil {
		return nil, nil, err
	}

	return bodyBuf.Bytes(), headers, nil
}

func (c deepinfraSpeechRecognitionCodec) Decode(
	params DecodeParams,
) (hftypes.SpeechRecognitionResponse, error) {
	return decodeOpenAITranscription(params)
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
	mapping, err := resolveMapping(
		params.Context,
		deepinfraProviderName,
		c.resolver,
		params.Mapping,
		TaskFeatureExtraction,
		params.Model,
	)
	if err != nil {
		return nil, nil, err
	}

	return encodeOpenAIEmbeddings(params.Request.Input, params.Request.ProviderParams, mapping)
}

func (c deepinfraFeatureExtractionCodec) Decode(
	params DecodeParams,
) (hftypes.FeatureExtractionPooledEmbedding, error) {
	embeddings, err := decodeOpenAIEmbeddings(params)
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
	mapping, err := resolveMapping(
		params.Context,
		deepinfraProviderName,
		c.resolver,
		params.Mapping,
		TaskFeatureExtraction,
		params.Model,
	)
	if err != nil {
		return nil, nil, err
	}

	return encodeOpenAIEmbeddings(params.Request.Inputs, params.Request.ProviderParams, mapping)
}

func (c deepinfraFeatureExtractionBatchCodec) Decode(
	params DecodeParams,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	return decodeOpenAIEmbeddings(params)
}
