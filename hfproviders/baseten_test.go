//go:build !integration

package hfproviders

// basetenChatOnlyCase returns the Baseten case for TestChatOnlyProviderContract.
func basetenChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        "baseten",
		newProvider: func() Provider { return NewBasetenProvider() },
	}
}
