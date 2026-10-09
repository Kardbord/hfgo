//go:build !integration

package hftypes_test

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestTextToImageRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.TextToImageRequest{
		Input: "a serene mountain landscape at sunset",
		Parameters: &hftypes.TextToImageParameters{
			GuidanceScale:     new(7.5),
			NegativePrompt:    new("blurry, low quality"),
			NumInferenceSteps: new(30),
			Width:             new(512),
			Height:            new(512),
			Scheduler:         new("DDIM"),
			Seed:              new(int64(42)),
		},
	}

	cloned := req.Clone()

	require.NotSame(t, req.Parameters, cloned.Parameters)
	cloned.Input = "changed"

	*cloned.Parameters.GuidanceScale = 1.0
	*cloned.Parameters.NegativePrompt = "changed"
	*cloned.Parameters.NumInferenceSteps = 1
	*cloned.Parameters.Width = 1
	*cloned.Parameters.Height = 1
	*cloned.Parameters.Scheduler = "changed"
	*cloned.Parameters.Seed = 1

	require.Equal(t, "a serene mountain landscape at sunset", req.Input)
	require.InEpsilon(t, 7.5, *req.Parameters.GuidanceScale, 0.001)
	require.Equal(t, "blurry, low quality", *req.Parameters.NegativePrompt)
	require.Equal(t, 30, *req.Parameters.NumInferenceSteps)
	require.Equal(t, 512, *req.Parameters.Width)
	require.Equal(t, 512, *req.Parameters.Height)
	require.Equal(t, "DDIM", *req.Parameters.Scheduler)
	require.Equal(t, int64(42), *req.Parameters.Seed)
}

func TestTextToImageParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.TextToImageParameters{
		GuidanceScale:     new(7.5),
		NegativePrompt:    new("blurry"),
		NumInferenceSteps: new(30),
		Width:             new(512),
		Height:            new(512),
		Scheduler:         new("DDIM"),
		Seed:              new(int64(42)),
	}

	cloned := params.Clone()

	*cloned.GuidanceScale = 1.0
	*cloned.NegativePrompt = "changed"
	*cloned.NumInferenceSteps = 1
	*cloned.Width = 1
	*cloned.Height = 1
	*cloned.Scheduler = "changed"
	*cloned.Seed = 1

	require.InEpsilon(t, 7.5, *params.GuidanceScale, 0.001)
	require.Equal(t, "blurry", *params.NegativePrompt)
	require.Equal(t, 30, *params.NumInferenceSteps)
	require.Equal(t, 512, *params.Width)
	require.Equal(t, 512, *params.Height)
	require.Equal(t, "DDIM", *params.Scheduler)
	require.Equal(t, int64(42), *params.Seed)
}

func TestTextToImageClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TextToImageRequest
	require.Empty(t, r.Clone())

	var p *hftypes.TextToImageParameters
	require.Empty(t, p.Clone())
}

// TextToImageResponse is assembled by the codec, not decoded from JSON, so its
// fields are excluded from JSON encoding. This test guards that intent.
func TestTextToImageResponse_JSONExcluded(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(hftypes.TextToImageResponse{
		Image:     []byte{0x89, 'P', 'N', 'G'},
		MediaType: "image/png",
	})
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(encoded))
}
