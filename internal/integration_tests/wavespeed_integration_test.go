//go:build integration

package integration_tests

import (
	"os"
	"testing"

	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

// newWaveSpeedProvider returns a [hfproviders.WaveSpeedProvider] backed by a
// Hub model resolver for live API tests. Callers gate the test on HF_TOKEN.
func newWaveSpeedProvider(t *testing.T) hfproviders.Provider {
	t.Helper()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubToken(os.Getenv("HF_TOKEN")),
	)

	provider, err := hfproviders.NewWaveSpeedProvider(resolver)
	require.NoError(t, err)

	return provider
}
