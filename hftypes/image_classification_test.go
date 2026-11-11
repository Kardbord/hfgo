//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestImageClassificationRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.ImageClassificationRequest{
		Input: "base64imgdata",
		Parameters: &hftypes.ImageClassificationParameters{
			FunctionToApply: new(hftypes.ImageClassificationFuncSoftmax),
			TopK:            new(5),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.FunctionToApply = hftypes.ImageClassificationFuncSigmoid
	*cloned.Parameters.TopK = 9
	cloned.Input = "changed"

	require.Equal(t, "base64imgdata", req.Input)
	require.Equal(t, hftypes.ImageClassificationFuncSoftmax, *req.Parameters.FunctionToApply)
	require.Equal(t, 5, *req.Parameters.TopK)
	require.Equal(t, "changed", cloned.Input)
	require.Equal(t, hftypes.ImageClassificationFuncSigmoid, *cloned.Parameters.FunctionToApply)
	require.Equal(t, 9, *cloned.Parameters.TopK)
}

func TestImageClassificationParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.ImageClassificationParameters{
		FunctionToApply: new(hftypes.ImageClassificationFuncSoftmax),
		TopK:            new(5),
	}

	cloned := params.Clone()

	*cloned.FunctionToApply = hftypes.ImageClassificationFuncNone
	*cloned.TopK = 1

	require.Equal(t, hftypes.ImageClassificationFuncSoftmax, *params.FunctionToApply)
	require.Equal(t, 5, *params.TopK)
	require.Equal(t, hftypes.ImageClassificationFuncNone, *cloned.FunctionToApply)
	require.Equal(t, 1, *cloned.TopK)
}

func TestImageClassificationClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.ImageClassificationRequest
	require.Empty(t, r.Clone())

	var p *hftypes.ImageClassificationParameters
	require.Empty(t, p.Clone())
}
