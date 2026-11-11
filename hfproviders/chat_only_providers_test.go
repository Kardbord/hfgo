//go:build !integration

package hfproviders

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// chatOnlyProviderCase describes one built-in chat-only partner provider:
// its router slug and constructor.
type chatOnlyProviderCase struct {
	slug        string
	newProvider func() Provider
}

// TestChatOnlyProviderContract pins the shared behavior of every built-in
// chat-only partner provider: identity equal to the router slug, the shared
// model-independent OpenAI-compatible chat endpoint and codecs (see
// [requireOpenAIChatPartnerEndpoint]), and exactly the chat task family. Add
// a case per new provider.
func TestChatOnlyProviderContract(t *testing.T) {
	t.Parallel()

	cases := []chatOnlyProviderCase{
		basetenChatOnlyCase(),
		cerebrasChatOnlyCase(),
		cohereChatOnlyCase(),
		featherlessAIChatOnlyCase(),
		fireworksAIChatOnlyCase(),
		groqChatOnlyCase(),
		novitaChatOnlyCase(),
		ovhcloudChatOnlyCase(),
		publicAIChatOnlyCase(),
		zaiChatOnlyCase(),
	}

	for _, tc := range cases {
		t.Run(tc.slug, func(t *testing.T) {
			t.Parallel()

			p := tc.newProvider()

			// Identity: Name and ProviderSuffix both equal the router slug.
			require.Equal(t, tc.slug, p.Name())
			require.Equal(t, tc.slug, p.ProviderSuffix())

			// Endpoint and codec wiring: the shared chat half of the partner
			// contract — the fixed OpenAI-compatible path with no mapping,
			// the model-independent endpoint, and the built-in JSON codecs.
			requireOpenAIChatPartnerEndpoint(t, p, tc.slug)

			// Task coverage: the provider advertises exactly the chat task
			// family and nothing else, mirroring the Hugging Face provider's
			// coverage assertions.
			_, err := AsProvider[SummarizationProvider](p)
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
