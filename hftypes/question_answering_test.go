//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestQuestionAnsweringRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.QuestionAnsweringRequest{
		Input: hftypes.QuestionAnsweringInput{
			Question: "What is the capital of France?",
			Context:  "France is a country in Europe. Its capital is Paris.",
		},
		Parameters: &hftypes.QuestionAnsweringParameters{
			TopK:                   new(3),
			DocStride:              new(128),
			MaxAnswerLen:           new(50),
			MaxSeqLen:              new(384),
			MaxQuestionLen:         new(64),
			HandleImpossibleAnswer: new(true),
			AlignToWords:           new(true),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.TopK = 5
	cloned.Parameters.HandleImpossibleAnswer = new(false)
	cloned.Input.Question = "changed"

	require.Equal(t, "What is the capital of France?", req.Input.Question)
	require.Equal(t, 3, *req.Parameters.TopK)
	require.True(t, *req.Parameters.HandleImpossibleAnswer)
	require.Equal(t, "changed", cloned.Input.Question)
	require.Equal(t, 5, *cloned.Parameters.TopK)
	require.False(t, *cloned.Parameters.HandleImpossibleAnswer)
}

func TestQuestionAnsweringParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.QuestionAnsweringParameters{
		TopK:                   new(3),
		DocStride:              new(128),
		MaxAnswerLen:           new(50),
		MaxSeqLen:              new(384),
		MaxQuestionLen:         new(64),
		HandleImpossibleAnswer: new(true),
		AlignToWords:           new(true),
	}

	cloned := params.Clone()

	*cloned.TopK = 5
	*cloned.DocStride = 256
	*cloned.MaxAnswerLen = 100
	*cloned.MaxSeqLen = 512
	*cloned.MaxQuestionLen = 128
	*cloned.HandleImpossibleAnswer = false
	*cloned.AlignToWords = false

	require.Equal(t, 3, *params.TopK)
	require.Equal(t, 128, *params.DocStride)
	require.Equal(t, 50, *params.MaxAnswerLen)
	require.Equal(t, 384, *params.MaxSeqLen)
	require.Equal(t, 64, *params.MaxQuestionLen)
	require.True(t, *params.HandleImpossibleAnswer)
	require.True(t, *params.AlignToWords)
	require.Equal(t, 5, *cloned.TopK)
	require.Equal(t, 256, *cloned.DocStride)
	require.Equal(t, 100, *cloned.MaxAnswerLen)
	require.Equal(t, 512, *cloned.MaxSeqLen)
	require.Equal(t, 128, *cloned.MaxQuestionLen)
	require.False(t, *cloned.HandleImpossibleAnswer)
	require.False(t, *cloned.AlignToWords)
}

func TestQuestionAnsweringClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.QuestionAnsweringRequest
	require.Empty(t, r.Clone())

	var p *hftypes.QuestionAnsweringParameters
	require.Empty(t, p.Clone())
}
