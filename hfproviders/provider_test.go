//go:build !integration

package hfproviders

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
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
	require.ErrorContains(t, err, "<nil> does not implement hfproviders.ChatProvider")
}

func TestAsProvider_ExactType(t *testing.T) {
	t.Parallel()

	// Casting to the interface the value itself implements succeeds too.
	p, err := AsProvider[Provider](minimalProvider{})
	require.NoError(t, err)
	require.Equal(t, "minimal", p.Name())
}
