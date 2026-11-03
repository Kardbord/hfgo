//go:build !integration

package hfproviders

// novitaChatOnlyCase returns the Novita case for TestChatOnlyProviderContract.
func novitaChatOnlyCase() chatOnlyProviderCase {
	return chatOnlyProviderCase{
		slug:        novitaProviderName,
		newProvider: func() Provider { return NewNovitaProvider() },
	}
}
