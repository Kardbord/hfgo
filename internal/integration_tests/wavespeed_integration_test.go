//go:build integration

package integration_tests

import (
	"os"
	"testing"

	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

// newWavespeedProvider returns a [hfproviders.WavespeedProvider] backed by a
// Hub model resolver for live API tests. It requires HF_TOKEN to be set.
func newWavespeedProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubToken(apiToken))

	provider, err := hfproviders.NewWavespeedProvider(resolver)
	require.NoError(t, err)

	return provider
}
