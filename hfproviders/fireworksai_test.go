//go:build !integration

package hfproviders

// fireworksAIChatOnlyCase returns the Fireworks AI case for
// TestChatOnlyProviderContract.
func fireworksAIChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "fireworks-ai",
		newProvider: func() Provider { return NewFireworksAIProvider() },
	}
}
