//go:build !integration

package hftypes

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestQuestionAnsweringRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := QuestionAnsweringRequest{
		Input: QuestionAnsweringInput{
			Question: "What is the capital of France?",
			Context:  "France is a country in Europe. Its capital is Paris.",
		},
		Parameters: &QuestionAnsweringParameters{
			TopK:                   testutils.Ptr(3),
			DocStride:              testutils.Ptr(128),
			MaxAnswerLen:           testutils.Ptr(50),
			MaxSeqLen:              testutils.Ptr(384),
			MaxQuestionLen:         testutils.Ptr(64),
			HandleImpossibleAnswer: testutils.Ptr(true),
			AlignToWords:           testutils.Ptr(true),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.TopK = 5
	cloned.Parameters.HandleImpossibleAnswer = testutils.Ptr(false)
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

	params := &QuestionAnsweringParameters{
		TopK:                   testutils.Ptr(3),
		DocStride:              testutils.Ptr(128),
		MaxAnswerLen:           testutils.Ptr(50),
		MaxSeqLen:              testutils.Ptr(384),
		MaxQuestionLen:         testutils.Ptr(64),
		HandleImpossibleAnswer: testutils.Ptr(true),
		AlignToWords:           testutils.Ptr(true),
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

	var r *QuestionAnsweringRequest
	require.Empty(t, r.Clone())

	var p *QuestionAnsweringParameters
	require.Empty(t, p.Clone())
}
