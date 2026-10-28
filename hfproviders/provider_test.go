//go:build !integration

package hfproviders

import (
	"context"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// minimalProvider implements only the core Provider interface.
type minimalProvider struct{}

func (minimalProvider) ProviderSuffix() string { return "" }

func (minimalProvider) Name() string { return "minimal" }

func TestAsProvider_Success(t *testing.T) {
	t.Parallel()

	chat, err := AsProvider[ChatProvider](NewHuggingFaceProvider())
	require.NoError(t, err)
	require.NotNil(t, chat)

	endpoint, err := chat.ChatEndpoint(EndpointParams{Model: "mistral-7b"})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
}

func TestAsProvider_WrongType(t *testing.T) {
	t.Parallel()

	_, err := AsProvider[SummarizationProvider](minimalProvider{})
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"minimal does not implement hfproviders.SummarizationProvider",
	)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

func TestAsProvider_NilProvider(t *testing.T) {
	t.Parallel()

	_, err := AsProvider[ChatProvider](nil)
	require.Error(t, err)
	require.ErrorContains(t, err, "provider must not be nil")
}

func TestAsProvider_TypedNilProvider(t *testing.T) {
	t.Parallel()

	// A typed-nil pointer satisfies interface != nil and its method set can
	// pass the cast, so it must be rejected explicitly before any value
	// receiver dereferences it.
	var p *HuggingFaceProvider
	_, err := AsProvider[ChatProvider](p)
	require.Error(t, err)
	require.ErrorContains(t, err, "provider must not be nil")

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

func TestAsProvider_ExactType(t *testing.T) {
	t.Parallel()

	// Casting to the interface the value itself implements succeeds too.
	p, err := AsProvider[Provider](minimalProvider{})
	require.NoError(t, err)
	require.Equal(t, "minimal", p.Name())
}

// subsetProvider serves only text classification and summarization by
// embedding the corresponding Hugging Face mixins.
type subsetProvider struct {
	HuggingFaceTextClassificationMixin
	HuggingFaceSummarizationMixin
}

func (subsetProvider) ProviderSuffix() string { return "" }

func (subsetProvider) Name() string { return "subset" }

func TestAsProvider_MixinSubset(t *testing.T) {
	t.Parallel()

	p := subsetProvider{}

	_, err := AsProvider[TextClassificationProvider](p)
	require.NoError(t, err)

	_, err = AsProvider[SummarizationProvider](p)
	require.NoError(t, err)

	_, err = AsProvider[ChatProvider](p)
	require.Error(t, err)
	require.ErrorContains(t, err, "subset does not implement hfproviders.ChatProvider")
}

// TestHuggingFaceProvider_TaskCoverage asserts the Hugging Face provider does
// not advertise tasks it does not serve. Text-to-video and image-to-image are
// served through partner providers only, e.g. [WavespeedProvider].
func TestHuggingFaceProvider_TaskCoverage(t *testing.T) {
	t.Parallel()

	p := NewHuggingFaceProvider()

	_, err := AsProvider[ImageToImageProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"huggingface does not implement hfproviders.ImageToImageProvider",
	)

	_, err = AsProvider[TextToVideoProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"huggingface does not implement hfproviders.TextToVideoProvider",
	)
}

// absolutePathProvider implements SummarizationProvider but returns an
// absolute endpoint path, which the transport must reject so the provider can
// never redirect a request — or its bearer token — away from the caller's
// configured host.
type absolutePathProvider struct{}

func (absolutePathProvider) ProviderSuffix() string { return "" }

func (absolutePathProvider) Name() string { return "absolute-path" }

func (absolutePathProvider) SummarizationEndpoint(_ EndpointParams) (Endpoint, error) {
	return Endpoint{Method: http.MethodPost, Path: "https://evil.example.com/steal"}, nil
}

func (absolutePathProvider) SummarizationCodec() SummarizationCodec {
	return JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{}
}

// TestProvider_AbsoluteEndpointPathIsRejected pins the security invariant from
// the provider side: a provider returning an absolute endpoint path fails with
// a configuration error before any request reaches the transport.
func TestProvider_AbsoluteEndpointPathIsRejected(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)
	client := testutils.NewMockHTTPClient(mt)

	p, err := AsProvider[SummarizationProvider](absolutePathProvider{})
	require.NoError(t, err)

	endpoint, err := p.SummarizationEndpoint(EndpointParams{Model: "any-model"})
	require.NoError(t, err)

	// Dispatch through the same transport the task pipeline uses.
	_, err = request.Do(
		context.Background(),
		request.Params{
			HTTPClient: &client,
			BaseURL:    "https://example.com",
		},
		endpoint.Method,
		endpoint.Path,
		nil,
	)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
	require.ErrorContains(t, err, "path must be relative")
	// The redirect is blocked before dispatch: no request reaches the wire.
	require.Nil(t, mt.LastRequest)
}
