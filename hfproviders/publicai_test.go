//go:build !integration

package hfproviders

// publicAIChatOnlyCase returns the Public AI case for
// TestChatOnlyProviderContract.
func publicAIChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "publicai",
		newProvider: func() Provider { return NewPublicAIProvider() },
		sampleModel: "speakleash/Bielik-11B-v3.0-Instruct",
	}
}
