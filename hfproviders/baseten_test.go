//go:build !integration

package hfproviders

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestBasetenProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	p := NewBasetenProvider()

	require.Equal(t, "baseten", p.Name())
	require.Equal(t, "baseten", p.ProviderSuffix())
}

// TestBasetenProvider_ChatEndpoint pins the endpoint wiring: chat completions
// are served from the fixed OpenAI-compatible path with no mapping attached,
// and the endpoint stays model-independent — an empty Model is legal.
func TestBasetenProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	p := NewBasetenProvider()

	endpoint, err := p.ChatEndpoint(EndpointParams{Model: "zai-org/GLM-5.3-Flash:baseten"})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
	require.Nil(t, endpoint.Mapping)

	endpoint, err = p.ChatEndpoint(EndpointParams{})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
}

// TestBasetenProvider_CodecWiring asserts the codec accessors return the
// built-in JSON codecs, pinning that Baseten reuses the default chat wire
// format rather than a provider-specific one.
func TestBasetenProvider_CodecWiring(t *testing.T) {
	t.Parallel()

	p := NewBasetenProvider()

	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
		p.ChatCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		p.ChatStreamCodec(),
	)
}

// TestBasetenProvider_TaskCoverage asserts the provider advertises exactly the
// chat task family and nothing else, mirroring the Hugging Face provider's
// coverage assertions.
func TestBasetenProvider_TaskCoverage(t *testing.T) {
	t.Parallel()

	p := NewBasetenProvider()

	_, err := AsProvider[ChatProvider](p)
	require.NoError(t, err)

	_, err = AsProvider[ChatStreamProvider](p)
	require.NoError(t, err)

	_, err = AsProvider[SummarizationProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"baseten does not implement hfproviders.SummarizationProvider",
	)

	_, err = AsProvider[TextToImageProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"baseten does not implement hfproviders.TextToImageProvider",
	)
}
