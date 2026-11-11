package hfproviders

import (
	"context"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hftypes"
)

// scalewayProviderName is the Scaleway provider slug: the model-suffix token
// the chat task pins routing with, and the Hub identifier the model resolver
// looks mappings up by.
const scalewayProviderName = "scaleway"

// scalewayEmbeddingsPath is the router-proxied embeddings endpoint. Single
// and batch requests share the same path.
const scalewayEmbeddingsPath = scalewayProviderName + "/v1/embeddings"

// ScalewayProvider serves chat completion and pooled feature-extraction
// tasks through the Hugging Face router.
//
// Chat (unary and streaming, including vision-language requests) reuses
// [HuggingFaceChatMixin] and pins routing by appending the router's
// ":scaleway" suffix to the model ID, exactly as the other chat partner
// providers do: the router receives the Hugging Face model ID plus suffix and
// performs provider mapping server-side. The suffix is a deterministic pin —
// the router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Scaleway does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin. This routes through the unified
// OpenAI-compatible chat endpoint rather than the provider-scoped path the
// official Hugging Face clients use for Scaleway, matching the convention
// shared by every other chat partner provider in this package.
//
// Feature extraction is served through Scaleway's OpenAI-compatible
// embeddings endpoint, proxied by the router under the "scaleway" path
// prefix, with Scaleway's own model identifiers in the request body. Callers
// configure the Hugging Face model ID they know; the pipeline mixin
// translates it to Scaleway's identifier at endpoint-construction time via
// its [ModelResolver], and the codecs reuse the resolution from
// [EncodeParams.Mapping]. The codec deviates from the official clients in one
// deliberate way: the hf-inference feature-extraction parameters the
// canonical request carries (e.g. normalize) are not forwarded, as Scaleway's
// embeddings API does not accept them — Scaleway-specific parameters such as
// encoding_format ride the request's ProviderParams field instead. The mapped
// model is applied to the payload last, so ProviderParams entries cannot
// override it.
//
// Only one mixin carries a resolver, so the promoted field
// ScalewayProvider.Resolver is unambiguous; both NewScalewayProvider and
// custom embeddings embed it directly. Contrast DeepInfraProvider, whose two
// resolver-bearing mixins leave the promoted field ambiguous.
//
// The zero value is safe to use: pipeline endpoint methods return a
// configuration [hferrors.SDKError] instead of dereferencing a nil resolver.
// As with the Hugging Face mixins, the Scaleway mixin can be embedded on its
// own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type ScalewayProvider struct {
	HuggingFaceChatMixin
	ScalewayFeatureExtractionMixin
}

// Compile-time assertions that ScalewayProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider                         = (*ScalewayProvider)(nil)
	_ ChatStreamProvider                   = (*ScalewayProvider)(nil)
	_ FeatureExtractionPooledProvider      = (*ScalewayProvider)(nil)
	_ FeatureExtractionPooledBatchProvider = (*ScalewayProvider)(nil)
)

// NewScalewayProvider creates a [ScalewayProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value ScalewayProvider is still safe to call; its
// pipeline endpoint methods return a configuration [hferrors.SDKError].
func NewScalewayProvider(resolver ModelResolver) (ScalewayProvider, error) {
	if resolver == nil {
		return ScalewayProvider{}, errNilResolver(scalewayProviderName)
	}

	return ScalewayProvider{
		HuggingFaceChatMixin: HuggingFaceChatMixin{},
		Resolver:             resolver,
	}, nil
}

// Name returns the name of the Scaleway provider for debug purposes.
func (p ScalewayProvider) Name() string {
	return scalewayProviderName
}

// ProviderSuffix returns the "scaleway" provider suffix for appending to the
// model selection in chat completion tasks. Pipeline tasks route via their
// endpoint paths and do not use the suffix.
func (p ScalewayProvider) ProviderSuffix() string {
	return scalewayProviderName
}

// ScalewayFeatureExtractionMixin provides the Scaleway endpoint and codecs
// for pooled feature extraction, including its batch variant. It is a mixin:
// embed it in a type that also implements Provider.Name and
// Provider.ProviderSuffix to serve only feature extraction. Its
// [ModelResolver] maps the configured Hugging Face model ID to Scaleway's
// provider model identifier at endpoint-construction time.
//
// Only pooled output is implemented: Scaleway's OpenAI-compatible embeddings
// endpoint returns one vector per input, never per-token embeddings, so this
// mixin does not satisfy the token-level feature-extraction provider
// interfaces.
type ScalewayFeatureExtractionMixin struct {
	// Resolver maps the configured Hugging Face model ID to Scaleway's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint methods are called.
	Resolver ModelResolver
}

// FeatureExtractionEndpoint resolves the configured Hugging Face model ID to
// Scaleway's identifier and returns the router-proxied path for the
// embeddings endpoint.
func (m ScalewayFeatureExtractionMixin) FeatureExtractionEndpoint(
	ctx context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return resolveScalewayEndpoint(
		ctx,
		m.Resolver,
		params,
		TaskFeatureExtraction,
		scalewayEmbeddingsPath,
	)
}

// FeatureExtractionBatchEndpoint resolves the configured Hugging Face model
// ID to Scaleway's identifier and returns the router-proxied path for the
// embeddings endpoint. Single and batch requests share the same path.
func (m ScalewayFeatureExtractionMixin) FeatureExtractionBatchEndpoint(
	ctx context.Context,
	params EndpointParams,
) (Endpoint, error) {
	return resolveScalewayEndpoint(
		ctx,
		m.Resolver,
		params,
		TaskFeatureExtraction,
		scalewayEmbeddingsPath,
	)
}

// FeatureExtractionPooledCodec returns the pooled codec for the single-input
// FeatureExtraction task, which encodes requests as Scaleway's OpenAI-style
// embeddings JSON body and decodes responses into a pooled embedding.
func (m ScalewayFeatureExtractionMixin) FeatureExtractionPooledCodec() FeatureExtractionPooledCodec {
	return scalewayFeatureExtractionCodec{resolver: m.Resolver}
}

// FeatureExtractionPooledBatchCodec returns the pooled codec for the
// FeatureExtractionBatch task, which encodes requests as Scaleway's
// OpenAI-style embeddings JSON body and decodes responses into one pooled
// embedding per input.
func (m ScalewayFeatureExtractionMixin) FeatureExtractionPooledBatchCodec() FeatureExtractionPooledBatchCodec {
	return scalewayFeatureExtractionBatchCodec{resolver: m.Resolver}
}

// resolveScalewayEndpoint resolves the configured Hugging Face model ID to
// Scaleway's provider model identifier and returns the router-proxied
// endpoint at the given fixed path. The provider slug given to the resolver
// is Scaleway's Hub identifier — the key the Hub's mapping data uses — which
// is a provider fact, not something the SDK derives.
func resolveScalewayEndpoint(
	ctx context.Context,
	resolver ModelResolver,
	params EndpointParams,
	task Task,
	path string,
) (Endpoint, error) {
	mapping, err := resolveMapping(
		ctx,
		scalewayProviderName,
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

// Compile-time assertions that the Scaleway codecs implement their task
// codecs.
var (
	_ FeatureExtractionPooledCodec      = (*scalewayFeatureExtractionCodec)(nil)
	_ FeatureExtractionPooledBatchCodec = (*scalewayFeatureExtractionBatchCodec)(nil)
)

// scalewayFeatureExtractionCodec is a [Codec] for the pooled output of the
// single-input FeatureExtraction task. It encodes canonical requests as
// Scaleway's OpenAI-style embeddings JSON body and decodes responses into a
// pooled embedding.
type scalewayFeatureExtractionCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c scalewayFeatureExtractionCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.FeatureExtractionRequest],
) (body []byte, headers http.Header, err error) {
	mapping, err := resolveMapping(
		ctx,
		scalewayProviderName,
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

func (c scalewayFeatureExtractionCodec) Decode(
	_ context.Context,
	params DecodeParams,
) (hftypes.FeatureExtractionPooledEmbedding, error) {
	embeddings, err := decodeOpenAIEmbeddings(params)
	if err != nil {
		return nil, err
	}

	return embeddings[0], nil
}

// scalewayFeatureExtractionBatchCodec is a [Codec] for the pooled output of
// the FeatureExtractionBatch task. It encodes canonical requests as
// Scaleway's OpenAI-style embeddings JSON body and decodes responses into one
// pooled embedding per input, in order.
type scalewayFeatureExtractionBatchCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

func (c scalewayFeatureExtractionBatchCodec) Encode(
	ctx context.Context,
	params EncodeParams[hftypes.FeatureExtractionBatchRequest],
) (body []byte, headers http.Header, err error) {
	mapping, err := resolveMapping(
		ctx,
		scalewayProviderName,
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

func (c scalewayFeatureExtractionBatchCodec) Decode(
	_ context.Context,
	params DecodeParams,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	return decodeOpenAIEmbeddings(params)
}
