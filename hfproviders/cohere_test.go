//go:build !integration

package hfproviders

// cohereChatOnlyCase returns the Cohere case for TestChatOnlyProviderContract.
func cohereChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "cohere",
		newProvider: func() Provider { return NewCohereProvider() },
		sampleModel: "CohereLabs/tiny-aya-global",
	}
}
