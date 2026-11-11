//go:build !integration

package hftypes_test

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestImageToImageRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.ImageToImageRequest{
		Input: "abc123",
		Parameters: &hftypes.ImageToImageParameters{
			Prompt:            new("a serene mountain landscape at sunset"),
			GuidanceScale:     new(7.5),
			NegativePrompt:    new("blurry, low quality"),
			NumInferenceSteps: new(30),
			TargetSize: &hftypes.ImageToImageTargetSize{
				Width:  new(512),
				Height: new(512),
			},
		},
		ProviderParams: map[string]any{"seed": 42},
	}

	cloned := req.Clone()

	require.NotSame(t, req.Parameters, cloned.Parameters)
	require.NotSame(t, req.Parameters.TargetSize, cloned.Parameters.TargetSize)
	require.NotSame(t, &req.ProviderParams, &cloned.ProviderParams)
	cloned.Input = "changed"

	*cloned.Parameters.Prompt = "changed"
	*cloned.Parameters.GuidanceScale = 1.0
	*cloned.Parameters.NegativePrompt = "changed"
	*cloned.Parameters.NumInferenceSteps = 1
	*cloned.Parameters.TargetSize.Width = 1
	*cloned.Parameters.TargetSize.Height = 1
	cloned.ProviderParams["steps"] = 5

	require.Equal(t, "abc123", req.Input)
	require.Equal(t, "a serene mountain landscape at sunset", *req.Parameters.Prompt)
	require.InEpsilon(t, 7.5, *req.Parameters.GuidanceScale, 0.001)
	require.Equal(t, "blurry, low quality", *req.Parameters.NegativePrompt)
	require.Equal(t, 30, *req.Parameters.NumInferenceSteps)
	require.Equal(t, 512, *req.Parameters.TargetSize.Width)
	require.Equal(t, 512, *req.Parameters.TargetSize.Height)
	require.Len(t, req.ProviderParams, 1)
	require.NotContains(t, req.ProviderParams, "steps")
}

func TestImageToImageParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.ImageToImageParameters{
		Prompt:            new("a serene mountain landscape at sunset"),
		GuidanceScale:     new(7.5),
		NegativePrompt:    new("blurry"),
		NumInferenceSteps: new(30),
		TargetSize: &hftypes.ImageToImageTargetSize{
			Width:  new(512),
			Height: new(512),
		},
	}

	cloned := params.Clone()

	*cloned.Prompt = "changed"
	*cloned.GuidanceScale = 1.0
	*cloned.NegativePrompt = "changed"
	*cloned.NumInferenceSteps = 1
	*cloned.TargetSize.Width = 1
	*cloned.TargetSize.Height = 1

	require.Equal(t, "a serene mountain landscape at sunset", *params.Prompt)
	require.InEpsilon(t, 7.5, *params.GuidanceScale, 0.001)
	require.Equal(t, "blurry", *params.NegativePrompt)
	require.Equal(t, 30, *params.NumInferenceSteps)
	require.Equal(t, 512, *params.TargetSize.Width)
	require.Equal(t, 512, *params.TargetSize.Height)
}

func TestImageToImageClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.ImageToImageRequest
	require.Empty(t, r.Clone())

	var p *hftypes.ImageToImageParameters
	require.Empty(t, p.Clone())

	var s *hftypes.ImageToImageTargetSize
	require.Empty(t, s.Clone())
}

// ImageToImageResponse is assembled by the codec, not decoded from JSON, so
// its fields are excluded from JSON encoding. This test guards that intent.
func TestImageToImageResponse_JSONExcluded(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(hftypes.ImageToImageResponse{
		Image:     []byte{0x89, 'P', 'N', 'G'},
		MediaType: "image/png",
	})
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(encoded))
}
