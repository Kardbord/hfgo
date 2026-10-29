//go:build !integration

package hfproviders

// ovhcloudChatOnlyCase returns the OVHcloud case for
// TestChatOnlyProviderContract.
func ovhcloudChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "ovhcloud",
		newProvider: func() Provider { return NewOVHcloudProvider() },
		sampleModel: "openai/gpt-oss-120b",
	}
}
