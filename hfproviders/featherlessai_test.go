//go:build !integration

package hfproviders

// featherlessAIChatOnlyCase returns the Featherless AI case for
// TestChatOnlyProviderContract.
func featherlessAIChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "featherless-ai",
		newProvider: func() Provider { return NewFeatherlessAIProvider() },
		sampleModel: "zai-org/GLM-5.3",
	}
}
