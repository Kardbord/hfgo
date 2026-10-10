//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestTextClassificationRequest_CloneDeep(t *testing.T) {
	t.Parallel()

	funcToApply := hftypes.TextClassificationFuncSoftmax
	topK := 5
	req := hftypes.TextClassificationRequest{
		Input: "test",
		Parameters: &hftypes.TextClassificationParameters{
			FunctionToApply: &funcToApply,
			TopK:            &topK,
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.FunctionToApply = hftypes.TextClassificationFuncSigmoid
	*cloned.Parameters.TopK = 3

	require.Equal(t, hftypes.TextClassificationFuncSoftmax, *req.Parameters.FunctionToApply)
	require.Equal(t, 5, *req.Parameters.TopK)
	require.Equal(t, hftypes.TextClassificationFuncSigmoid, *cloned.Parameters.FunctionToApply)
	require.Equal(t, 3, *cloned.Parameters.TopK)
}

func TestTextClassificationParameters_CloneDeep(t *testing.T) {
	t.Parallel()

	funcToApply := hftypes.TextClassificationFuncNone
	topK := 1
	params := hftypes.TextClassificationParameters{
		FunctionToApply: &funcToApply,
		TopK:            &topK,
	}

	cloned := params.Clone()

	*cloned.FunctionToApply = hftypes.TextClassificationFuncSoftmax
	*cloned.TopK = 5

	require.Equal(t, hftypes.TextClassificationFuncNone, *params.FunctionToApply)
	require.Equal(t, 1, *params.TopK)
	require.Equal(t, hftypes.TextClassificationFuncSoftmax, *cloned.FunctionToApply)
	require.Equal(t, 5, *cloned.TopK)
}

func TestTextClassificationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var req *hftypes.TextClassificationRequest
	require.Empty(t, req.Clone())

	var params *hftypes.TextClassificationParameters
	require.Empty(t, params.Clone())
}
