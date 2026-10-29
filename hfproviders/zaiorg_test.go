//go:build !integration

package hfproviders

// zaiChatOnlyCase returns the Z.ai case for TestChatOnlyProviderContract.
func zaiChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "zai-org",
		newProvider: func() Provider { return NewZAIProvider() },
		sampleModel: "zai-org/GLM-5.3",
	}
}
