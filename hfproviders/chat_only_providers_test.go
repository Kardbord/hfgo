//go:build !integration

package hfproviders

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

// chatOnlyProviderCase describes one built-in chat-only partner provider:
// its router slug, constructor, and a model the provider serves.
type chatOnlyProviderCase struct {
	slug        string
	newProvider func() Provider
	sampleModel string
}

// TestChatOnlyProviderContract pins the shared behavior of every built-in
// chat-only partner provider: identity equal to the router slug, the
// model-independent fixed chat endpoint with no mapping attached, the built-in
// JSON chat codecs, and exactly the chat task family. Add a case per new
// provider.
func TestChatOnlyProviderContract(t *testing.T) {
	t.Parallel()

	cases := []chatOnlyProviderCase{
		basetenChatOnlyCase(),
		cerebrasChatOnlyCase(),
		cohereChatOnlyCase(),
		featherlessAIChatOnlyCase(),
		fireworksChatOnlyCase(),
		groqChatOnlyCase(),
	}

	for _, tc := range cases {
		t.Run(tc.slug, func(t *testing.T) {
			t.Parallel()

			p := tc.newProvider()

			chatProv, err := AsProvider[ChatProvider](p)
			require.NoError(t, err)

			streamProv, err := AsProvider[ChatStreamProvider](p)
			require.NoError(t, err)

			// Identity: Name and ProviderSuffix both equal the router slug.
			require.Equal(t, tc.slug, p.Name())
			require.Equal(t, tc.slug, p.ProviderSuffix())

			// Endpoint wiring: chat completions are served from the fixed
			// OpenAI-compatible path with no mapping attached, and the
			// endpoint stays model-independent — an empty Model is legal.
			endpoint, err := chatProv.ChatEndpoint(
				EndpointParams{Model: tc.sampleModel + ":" + tc.slug},
			)
			require.NoError(t, err)
			require.Equal(t, "v1/chat/completions", endpoint.Path)
			require.Nil(t, endpoint.Mapping)

			endpoint, err = chatProv.ChatEndpoint(EndpointParams{})
			require.NoError(t, err)
			require.Equal(t, "v1/chat/completions", endpoint.Path)

			// Codec wiring: the built-in JSON codecs, pinning that the
			// provider reuses the default chat wire format rather than a
			// provider-specific one.
			require.IsType(
				t,
				JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
				chatProv.ChatCodec(),
			)
			require.IsType(
				t,
				JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
				streamProv.ChatStreamCodec(),
			)

			// Task coverage: the provider advertises exactly the chat task
			// family and nothing else, mirroring the Hugging Face provider's
			// coverage assertions.
			_, err = AsProvider[SummarizationProvider](p)
			require.Error(t, err)
			require.ErrorContains(
				t,
				err,
				tc.slug+" does not implement hfproviders.SummarizationProvider",
			)

			_, err = AsProvider[TextToImageProvider](p)
			require.Error(t, err)
			require.ErrorContains(
				t,
				err,
				tc.slug+" does not implement hfproviders.TextToImageProvider",
			)
		})
	}
}
