//go:build !integration

package hfproviders

// cerebrasChatOnlyCase returns the Cerebras case for TestChatOnlyProviderContract.
func cerebrasChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "cerebras",
		newProvider: func() Provider { return NewCerebrasProvider() },
		sampleModel: "openai/gpt-oss-120b",
	}
}
