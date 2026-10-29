//go:build !integration

package hfproviders

// groqChatOnlyCase returns the Groq case for TestChatOnlyProviderContract.
func groqChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "groq",
		newProvider: func() Provider { return NewGroqProvider() },
		sampleModel: "openai/gpt-oss-120b",
	}
}
