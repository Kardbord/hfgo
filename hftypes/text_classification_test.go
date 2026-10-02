//go:build !integration

package hftypes

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTextClassificationRequest_CloneDeep(t *testing.T) {
	t.Parallel()

	funcToApply := TextClassificationFuncSoftmax
	topK := 5
	req := TextClassificationRequest{
		Input: "test",
		Parameters: &TextClassificationParameters{
			FunctionToApply: &funcToApply,
			TopK:            &topK,
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.FunctionToApply = TextClassificationFuncSigmoid
	*cloned.Parameters.TopK = 3

	require.Equal(t, TextClassificationFuncSoftmax, *req.Parameters.FunctionToApply)
	require.Equal(t, 5, *req.Parameters.TopK)
	require.Equal(t, TextClassificationFuncSigmoid, *cloned.Parameters.FunctionToApply)
	require.Equal(t, 3, *cloned.Parameters.TopK)
}

func TestTextClassificationParameters_CloneDeep(t *testing.T) {
	t.Parallel()

	funcToApply := TextClassificationFuncNone
	topK := 1
	params := TextClassificationParameters{
		FunctionToApply: &funcToApply,
		TopK:            &topK,
	}

	cloned := params.Clone()

	*cloned.FunctionToApply = TextClassificationFuncSoftmax
	*cloned.TopK = 5

	require.Equal(t, TextClassificationFuncNone, *params.FunctionToApply)
	require.Equal(t, 1, *params.TopK)
	require.Equal(t, TextClassificationFuncSoftmax, *cloned.FunctionToApply)
	require.Equal(t, 5, *cloned.TopK)
}

func TestTextClassificationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var req *TextClassificationRequest
	require.Empty(t, req.Clone())

	var params *TextClassificationParameters
	require.Empty(t, params.Clone())
}
