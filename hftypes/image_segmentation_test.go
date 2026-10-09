//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestImageSegmentationRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.ImageSegmentationRequest{
		Input: "base64imgdata",
		Parameters: &hftypes.ImageSegmentationParameters{
			MaskThreshold:            new(0.5),
			OverlapMaskAreaThreshold: new(0.25),
			Subtask:                  hftypes.ImageSegmentationSubtaskSemantic,
			Threshold:                new(0.7),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.MaskThreshold = 0.9
	*cloned.Parameters.OverlapMaskAreaThreshold = 0.9
	*cloned.Parameters.Threshold = 0.9
	cloned.Parameters.Subtask = hftypes.ImageSegmentationSubtaskInstance
	cloned.Input = "changed"

	require.Equal(t, "base64imgdata", req.Input)
	require.InEpsilon(t, 0.5, *req.Parameters.MaskThreshold, 0.001)
	require.InEpsilon(t, 0.25, *req.Parameters.OverlapMaskAreaThreshold, 0.001)
	require.InEpsilon(t, 0.7, *req.Parameters.Threshold, 0.001)
	require.Equal(t, hftypes.ImageSegmentationSubtaskSemantic, req.Parameters.Subtask)
	require.InEpsilon(t, 0.9, *cloned.Parameters.MaskThreshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.Parameters.OverlapMaskAreaThreshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.Parameters.Threshold, 0.001)
	require.Equal(t, hftypes.ImageSegmentationSubtaskInstance, cloned.Parameters.Subtask)
	require.Equal(t, "changed", cloned.Input)
}

func TestImageSegmentationParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.ImageSegmentationParameters{
		MaskThreshold:            new(0.5),
		OverlapMaskAreaThreshold: new(0.25),
		Subtask:                  hftypes.ImageSegmentationSubtaskPanoptic,
		Threshold:                new(0.7),
	}

	cloned := params.Clone()

	*cloned.MaskThreshold = 0.9
	*cloned.OverlapMaskAreaThreshold = 0.9
	*cloned.Threshold = 0.9
	cloned.Subtask = hftypes.ImageSegmentationSubtaskInstance

	require.InEpsilon(t, 0.5, *params.MaskThreshold, 0.001)
	require.InEpsilon(t, 0.25, *params.OverlapMaskAreaThreshold, 0.001)
	require.InEpsilon(t, 0.7, *params.Threshold, 0.001)
	require.Equal(t, hftypes.ImageSegmentationSubtaskPanoptic, params.Subtask)
	require.InEpsilon(t, 0.9, *cloned.MaskThreshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.OverlapMaskAreaThreshold, 0.001)
	require.InEpsilon(t, 0.9, *cloned.Threshold, 0.001)
	require.Equal(t, hftypes.ImageSegmentationSubtaskInstance, cloned.Subtask)
}

func TestImageSegmentationClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.ImageSegmentationRequest
	require.Empty(t, r.Clone())

	var p *hftypes.ImageSegmentationParameters
	require.Empty(t, p.Clone())
}
