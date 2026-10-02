//go:build !integration

package task

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

type mockProvider struct {
	hfproviders.DefaultCodec

	name string
}

func (p mockProvider) Endpoint(_ hfproviders.Task, _ string) (string, error) {
	return "", nil
}

func (p mockProvider) ProviderSuffix() string {
	return p.name
}

func TestApplyProvider(t *testing.T) {
	t.Parallel()

	t.Run("applies provider suffix", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		provider := mockProvider{name: "sambanova"}
		want := "mistral-7b:sambanova"
		got := applyProvider(model, provider)
		require.NotNil(t, got)
		require.Equal(t, want, *got)
	})

	t.Run("ignores provider with empty suffix", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		got := applyProvider(model, hfproviders.HuggingFaceProvider{})
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("returns nil when model is nil", func(t *testing.T) {
		got := applyProvider(nil, mockProvider{name: "sambanova"})
		require.Nil(t, got)
	})
}

func TestResolveModel(t *testing.T) {
	t.Parallel()

	t.Run("uses request model when provided", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: testutils.Ptr("request-model")}
		resolveModel(payload, hfopts.NewOptions().With(hfopts.WithModel("client-model")))
		require.NotNil(t, payload.Model)
		require.Equal(t, "request-model", *payload.Model)
	})

	t.Run("uses options model when request model is nil", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		resolveModel(payload, hfopts.NewOptions().With(hfopts.WithModel("opts-model")))
		require.NotNil(t, payload.Model)
		require.Equal(t, "opts-model", *payload.Model)
	})
}
