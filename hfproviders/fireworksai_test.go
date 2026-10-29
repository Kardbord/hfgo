//go:build !integration

package hfproviders

// fireworksChatOnlyCase returns the Fireworks case for
// TestChatOnlyProviderContract.
func fireworksChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "fireworks-ai",
		newProvider: func() Provider { return NewFireworksProvider() },
		sampleModel: "zai-org/GLM-5.3",
	}
}
