//go:build integration

package integration_tests

import (
	"os"
	"testing"

	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

// newWavespeedProvider returns a [hfproviders.WavespeedProvider] backed by a
// Hub model resolver for live API tests. Callers gate the test on HF_TOKEN.
func newWavespeedProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
	)

	provider, err := hfproviders.NewWavespeedProvider(resolver)
	require.NoError(t, err)

	return provider
}
