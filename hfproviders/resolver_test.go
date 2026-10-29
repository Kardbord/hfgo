//go:build !integration

package hfproviders_test

import (
	"context"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// resolveParams is a helper for building [hfproviders.ResolveModelParams].
func resolveParams(
	provider string,
	task hfproviders.Task,
	model string,
) hfproviders.ResolveModelParams {
	return hfproviders.ResolveModelParams{
		Provider: provider,
		Task:     task,
		Model:    model,
	}
}

func TestStaticModelResolver_ResolvesPinnedMapping(t *testing.T) {
	t.Parallel()

	mappings := map[hfproviders.StaticMapping]hfproviders.ModelMapping{
		{Provider: "fal-ai", Task: "text-to-image", Model: "black-forest-labs/FLUX.1-dev"}: {
			ProviderModelID: "fal-ai/flux/dev",
		},
	}
	resolver := hfproviders.NewStaticModelResolver(mappings)

	mapping, err := resolver.ResolveModel(context.Background(), resolveParams(
		"fal-ai", "text-to-image", "black-forest-labs/FLUX.1-dev",
	))
	require.NoError(t, err)
	require.Equal(t, hfproviders.ModelMapping{
		ProviderModelID: "fal-ai/flux/dev",
	}, mapping)
}

func TestStaticModelResolver_CarriesAdapterMetadata(t *testing.T) {
	t.Parallel()

	mappings := map[hfproviders.StaticMapping]hfproviders.ModelMapping{
		{
			Provider: "wavespeed",
			Task:     "text-to-video",
			Model:    "ZhengmingYu/DMAD",
		}: {
			ProviderModelID:    "wavespeed-ai/minimax-h3/text-to-video-lora",
			Adapter:            "lora",
			AdapterWeightsPath: "minimax_h3/dmad.safetensors",
			Status:             "live",
		},
	}
	resolver := hfproviders.NewStaticModelResolver(mappings)

	mapping, err := resolver.ResolveModel(context.Background(), resolveParams(
		"wavespeed", "text-to-video", "ZhengmingYu/DMAD",
	))
	require.NoError(t, err)
	require.Equal(t, hfproviders.ModelMapping{
		ProviderModelID:    "wavespeed-ai/minimax-h3/text-to-video-lora",
		Adapter:            "lora",
		AdapterWeightsPath: "minimax_h3/dmad.safetensors",
		Status:             "live",
	}, mapping)
}

func TestStaticModelResolver_ExactKeyMatch(t *testing.T) {
	t.Parallel()

	mappings := map[hfproviders.StaticMapping]hfproviders.ModelMapping{
		{
			Provider: "fal-ai",
			Task:     "text-to-image",
			Model:    "black-forest-labs/FLUX.1-dev",
		}: {ProviderModelID: "fal-ai/flux/dev"},
	}
	resolver := hfproviders.NewStaticModelResolver(mappings)

	cases := []struct {
		name     string
		provider string
		task     hfproviders.Task
		model    string
	}{
		{
			name:     "different provider",
			provider: "replicate",
			task:     "text-to-image",
			model:    "black-forest-labs/FLUX.1-dev",
		},
		{
			name:     "different task",
			provider: "fal-ai",
			task:     "text-generation",
			model:    "black-forest-labs/FLUX.1-dev",
		},
		{
			name:     "different model",
			provider: "fal-ai",
			task:     "text-to-image",
			model:    "stabilityai/stable-diffusion-xl",
		},
		{
			name:     "empty provider",
			provider: "",
			task:     "text-to-image",
			model:    "black-forest-labs/FLUX.1-dev",
		},
		{
			name:     "empty task",
			provider: "fal-ai",
			task:     "",
			model:    "black-forest-labs/FLUX.1-dev",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := resolver.ResolveModel(context.Background(), resolveParams(
				tc.provider, tc.task, tc.model,
			))
			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		})
	}
}

func TestStaticModelResolver_EmptyMappingFailsAllLookups(t *testing.T) {
	t.Parallel()

	resolver := hfproviders.NewStaticModelResolver(nil)

	_, err := resolver.ResolveModel(context.Background(), resolveParams(
		"fal-ai", "text-to-image", "black-forest-labs/FLUX.1-dev",
	))
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestStaticModelResolver_ErrorMessageNamesAllInputs(t *testing.T) {
	t.Parallel()

	resolver := hfproviders.NewStaticModelResolver(nil)

	_, err := resolver.ResolveModel(context.Background(), resolveParams(
		"fal-ai", "text-to-image", "black-forest-labs/FLUX.1-dev",
	))
	require.Error(t, err)
	require.Contains(t, err.Error(), "fal-ai")
	require.Contains(t, err.Error(), "text-to-image")
	require.Contains(t, err.Error(), "black-forest-labs/FLUX.1-dev")
}
