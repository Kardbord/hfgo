//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestObjectDetectionRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.ObjectDetectionRequest{
		Input: "base64imgdata",
		Parameters: &hftypes.ObjectDetectionParameters{
			Threshold: new(0.5),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.Threshold = 0.9
	cloned.Input = "changed"

	require.Equal(t, "base64imgdata", req.Input)
	require.InEpsilon(t, 0.5, *req.Parameters.Threshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.Parameters.Threshold, 0.001)
	require.Equal(t, "changed", cloned.Input)
}

func TestObjectDetectionParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.ObjectDetectionParameters{
		Threshold: new(0.5),
	}

	cloned := params.Clone()

	*cloned.Threshold = 0.9

	require.InEpsilon(t, 0.5, *params.Threshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.Threshold, 0.001)
}

func TestObjectDetectionClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.ObjectDetectionRequest
	require.Empty(t, r.Clone())

	var p *hftypes.ObjectDetectionParameters
	require.Empty(t, p.Clone())
}
