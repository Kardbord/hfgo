//go:build integration

package integration_tests

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

const fluxModel = "black-forest-labs/FLUX.1-dev"

// TestHubModelResolver_LiveHub resolves a model through the real Hub's
// inferenceProviderMapping data, exercising the full fetch and decode path
// against the live response shape. It calls only the public Hub metadata
// endpoint — no inference request is made.
// This test requires the HF_TOKEN environment variable to be set.
func TestHubModelResolver_LiveHub(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubToken(apiToken))

	id, err := resolver.ResolveModel(hfproviders.ResolveModelParams{
		Context:  ctx,
		Provider: "fal-ai",
		Task:     "text-to-image",
		Model:    fluxModel,
	})
	require.NoError(t, err)
	require.Equal(t, "fal-ai/flux/dev", id)
}

// TestHubModelResolver_LiveHubAbsentProvider checks that a provider with no
// mapping for the model surfaces a configuration error against the live Hub.
// This test requires the HF_TOKEN environment variable to be set.
func TestHubModelResolver_LiveHubAbsentProvider(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubToken(apiToken))

	_, err := resolver.ResolveModel(hfproviders.ResolveModelParams{
		Context:  ctx,
		Provider: "nonexistent-provider",
		Task:     "text-to-image",
		Model:    fluxModel,
	})
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}
