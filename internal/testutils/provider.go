package testutils

import "github.com/Kardbord/hfgo/v4/hfproviders"

// MockProvider is a reusable hfproviders.Provider implementation for tests.
// It embeds [hfproviders.HuggingFaceEndpoints] and
// [hfproviders.HuggingFaceCodecs], so it satisfies every per-task provider
// interface with Hugging Face defaults. Override individual behavior by
// embedding MockProvider in a local struct and shadowing methods.
type MockProvider struct {
	hfproviders.HuggingFaceEndpoints
	hfproviders.HuggingFaceCodecs

	mockName   string
	mockSuffix string
}

// NewMockProvider returns a MockProvider with the given debug name and model
// routing suffix.
func NewMockProvider(name, suffix string) MockProvider {
	return MockProvider{mockName: name, mockSuffix: suffix}
}

// Name returns the configured provider name.
func (p MockProvider) Name() string {
	return p.mockName
}

// ProviderSuffix returns the configured model routing suffix.
func (p MockProvider) ProviderSuffix() string {
	return p.mockSuffix
}
