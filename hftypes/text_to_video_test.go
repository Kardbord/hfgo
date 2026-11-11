//go:build !integration

package hftypes_test

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestTextToVideoRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.TextToVideoRequest{
		Input: "a serene mountain landscape at sunset",
		Parameters: &hftypes.TextToVideoParameters{
			NumFrames:         new(81),
			GuidanceScale:     new(7.5),
			NegativePrompt:    &[]string{"blurry", "low quality"},
			NumInferenceSteps: new(30),
			Seed:              new(int64(42)),
		},
		ProviderParams: map[string]any{"width": 512},
	}

	cloned := req.Clone()

	require.NotSame(t, req.Parameters, cloned.Parameters)
	require.NotSame(t, &req.ProviderParams, &cloned.ProviderParams)
	cloned.Input = "changed"

	*cloned.Parameters.NumFrames = 1
	*cloned.Parameters.GuidanceScale = 1.0
	(*cloned.Parameters.NegativePrompt)[0] = "changed"
	*cloned.Parameters.NumInferenceSteps = 1
	*cloned.Parameters.Seed = 1
	cloned.ProviderParams["height"] = 256

	require.Equal(t, "a serene mountain landscape at sunset", req.Input)
	require.Equal(t, 81, *req.Parameters.NumFrames)
	require.InEpsilon(t, 7.5, *req.Parameters.GuidanceScale, 0.001)
	require.Equal(t, "blurry", (*req.Parameters.NegativePrompt)[0])
	require.Equal(t, "low quality", (*req.Parameters.NegativePrompt)[1])
	require.Equal(t, 30, *req.Parameters.NumInferenceSteps)
	require.Equal(t, int64(42), *req.Parameters.Seed)
	require.Len(t, req.ProviderParams, 1)
	require.NotContains(t, req.ProviderParams, "height")
}

func TestTextToVideoParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.TextToVideoParameters{
		NumFrames:         new(81),
		GuidanceScale:     new(7.5),
		NegativePrompt:    &[]string{"blurry"},
		NumInferenceSteps: new(30),
		Seed:              new(int64(42)),
	}

	cloned := params.Clone()

	*cloned.NumFrames = 1
	*cloned.GuidanceScale = 1.0
	(*cloned.NegativePrompt)[0] = "changed"
	*cloned.NumInferenceSteps = 1
	*cloned.Seed = 1

	require.Equal(t, 81, *params.NumFrames)
	require.InEpsilon(t, 7.5, *params.GuidanceScale, 0.001)
	require.Equal(t, "blurry", (*params.NegativePrompt)[0])
	require.Equal(t, 30, *params.NumInferenceSteps)
	require.Equal(t, int64(42), *params.Seed)
}

func TestTextToVideoClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TextToVideoRequest
	require.Empty(t, r.Clone())

	var p *hftypes.TextToVideoParameters
	require.Empty(t, p.Clone())
}

// TextToVideoResponse is assembled by the codec, not decoded from JSON, so its
// fields are excluded from JSON encoding. This test guards that intent.
func TestTextToVideoResponse_JSONExcluded(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(hftypes.TextToVideoResponse{
		Video:     []byte{0x1A, 0x45, 0xDF, 0xA3},
		MediaType: "video/webm",
	})
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(encoded))
}
