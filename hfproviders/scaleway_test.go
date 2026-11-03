//go:build !integration

package hfproviders

import (
	"context"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const scalewayEmbedModel = "Qwen/Qwen3-Embedding-8B"

// newScalewayProvider builds a ScalewayProvider backed by a static resolver
// covering the test models.
func newScalewayProvider(t *testing.T) ScalewayProvider {
	t.Helper()

	provider, err := NewScalewayProvider(NewStaticModelResolver(map[StaticMapping]ModelMapping{
		{
			Provider: scalewayProviderName,
			Task:     TaskFeatureExtraction,
			Model:    scalewayEmbedModel,
		}: {ProviderModelID: "qwen3-embedding-8b"},
	}))
	require.NoError(t, err)

	return provider
}

func TestScalewayProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	provider := newScalewayProvider(t)
	require.Equal(t, "scaleway", provider.Name())
	require.Equal(t, "scaleway", provider.ProviderSuffix())
}

func TestNewScalewayProvider_NilResolver(t *testing.T) {
	t.Parallel()

	provider, err := NewScalewayProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, ScalewayProvider{}, provider)
}

// TestScalewayProvider_ZeroValueIsSafe verifies the zero-value provider's
// pipeline endpoint methods and codecs fail with configuration errors
// instead of dereferencing a nil resolver.
func TestScalewayProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var provider ScalewayProvider

	_, err := provider.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = provider.FeatureExtractionBatchEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	var featureMixin ScalewayFeatureExtractionMixin

	_, _, err = featureMixin.FeatureExtractionPooledCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, _, err = featureMixin.FeatureExtractionPooledBatchCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionBatchRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestScalewayProvider_Contract pins the task interfaces ScalewayProvider
// serves and the ones it must reject.
func TestScalewayProvider_Contract(t *testing.T) {
	t.Parallel()

	provider := newScalewayProvider(t)

	requireServedTaskInterfaces(t, provider, []func(Provider) error{
		func(p Provider) error {
			_, err := AsProvider[ChatProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[ChatStreamProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[FeatureExtractionPooledProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[FeatureExtractionPooledBatchProvider](p)

			return err
		},
	})

	requireMissingTaskInterfaces(t, provider, scalewayProviderName, []taskCast{
		{
			typeName: "hfproviders.SpeechRecognitionProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SpeechRecognitionProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.SummarizationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SummarizationProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TextToImageProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextToImageProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.FeatureExtractionTokenProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[FeatureExtractionTokenProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.FeatureExtractionTokenBatchProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[FeatureExtractionTokenBatchProvider](p)

				return err
			},
		},
	})
}

func TestScalewayProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	provider := newScalewayProvider(t)

	requireOpenAIChatPartnerEndpoint(t, provider, scalewayProviderName)
}

func TestScalewayProvider_FeatureExtractionEndpoints(t *testing.T) {
	t.Parallel()

	resolver := &recordingResolver{
		mapping: ModelMapping{
			ProviderModelID: "qwen3-embedding-8b",
			Status:          "live",
		},
	}
	provider, err := NewScalewayProvider(resolver)
	require.NoError(t, err)

	requireOpenAIEmbeddingsEndpoints(
		t,
		provider,
		scalewayEmbedModel,
		"scaleway/v1/embeddings",
		"qwen3-embedding-8b",
	)

	require.Equal(t, ResolveModelParams{
		Provider: "scaleway",
		Task:     TaskFeatureExtraction,
		Model:    scalewayEmbedModel,
	}, resolver.got)
}

func TestScalewayFeatureExtractionCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newScalewayProvider(t)

	assertOpenAIEmbeddingsCodecEncode(
		t,
		provider.FeatureExtractionPooledCodec(),
		scalewayEmbedModel,
		"qwen3-embedding-8b",
		func(resolver ModelResolver) FeatureExtractionPooledCodec {
			return ScalewayFeatureExtractionMixin{
				Resolver: resolver,
			}.FeatureExtractionPooledCodec()
		},
	)
}

func TestScalewayFeatureExtractionBatchCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newScalewayProvider(t)
	codec := provider.FeatureExtractionPooledBatchCodec()

	body, _, err := codec.Encode(EncodeParams[hftypes.FeatureExtractionBatchRequest]{
		Context: context.Background(),
		Request: hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{"first input", "second input"},
		},
		Model: scalewayEmbedModel,
	})
	require.NoError(t, err)
	require.JSONEq(
		t,
		`{"input":["first input","second input"],"model":"qwen3-embedding-8b"}`,
		string(body),
	)
}

// scalewaySubsetProvider serves only feature extraction by embedding the
// corresponding Scaleway mixin with a resolver set directly, without going
// through NewScalewayProvider.
type scalewaySubsetProvider struct {
	ScalewayFeatureExtractionMixin
}

func (scalewaySubsetProvider) ProviderSuffix() string { return scalewayProviderName }

func (scalewaySubsetProvider) Name() string { return "subset" }

func TestScalewayMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	provider := scalewaySubsetProvider{
		ScalewayFeatureExtractionMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: scalewayProviderName,
					Task:     TaskFeatureExtraction,
					Model:    scalewayEmbedModel,
				}: {ProviderModelID: "qwen3-embedding-8b"},
			}),
		},
	}

	pooled, err := AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	endpoint, err := pooled.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   scalewayEmbedModel,
	})
	require.NoError(t, err)
	require.Equal(t, "scaleway/v1/embeddings", endpoint.Path)

	_, err = AsProvider[ChatProvider](provider)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.ChatProvider",
	)

	_, err = AsProvider[SpeechRecognitionProvider](provider)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.SpeechRecognitionProvider",
	)
}

// TestScalewayProvider_Tasks is a compile-time pin that the constructor-built
// provider implements exactly the documented task interfaces; the casts run
// through AsProvider like the dispatch layer does.
func TestScalewayProvider_Tasks(t *testing.T) {
	t.Parallel()

	provider, err := NewScalewayProvider(NewStaticModelResolver(nil))
	require.NoError(t, err)

	_, err = AsProvider[ChatProvider](provider)
	require.NoError(t, err)

	_, err = AsProvider[ChatStreamProvider](provider)
	require.NoError(t, err)

	_, err = AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	_, err = AsProvider[FeatureExtractionPooledBatchProvider](provider)
	require.NoError(t, err)
}

// TestScalewayEndpoint_ResolverErrorsPropagate pins that a resolver error
// (not just a nil resolver) surfaces from the endpoint methods.
func TestScalewayEndpoint_ResolverErrorsPropagate(t *testing.T) {
	t.Parallel()

	provider, err := NewScalewayProvider(failingResolver{})
	require.NoError(t, err)

	feature, err := AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	_, err = feature.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   scalewayEmbedModel,
	})
	require.ErrorContains(t, err, "resolver outage")

	batch, err := AsProvider[FeatureExtractionPooledBatchProvider](provider)
	require.NoError(t, err)

	_, err = batch.FeatureExtractionBatchEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   scalewayEmbedModel,
	})
	require.ErrorContains(t, err, "resolver outage")
}

// TestScalewayCodec_ResolverErrorsPropagate pins that a resolver error
// surfaces from the codecs when used standalone (no mapping supplied).
func TestScalewayCodec_ResolverErrorsPropagate(t *testing.T) {
	t.Parallel()

	provider, err := NewScalewayProvider(failingResolver{})
	require.NoError(t, err)

	_, _, err = provider.FeatureExtractionPooledCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionRequest]{
			Context: context.Background(),
			Request: hftypes.FeatureExtractionRequest{Input: "hello"},
			Model:   scalewayEmbedModel,
		})
	require.ErrorContains(t, err, "resolver outage")

	_, _, err = provider.FeatureExtractionPooledBatchCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionBatchRequest]{
			Context: context.Background(),
			Request: hftypes.FeatureExtractionBatchRequest{Inputs: []string{"hello"}},
			Model:   scalewayEmbedModel,
		})
	require.ErrorContains(t, err, "resolver outage")
}
