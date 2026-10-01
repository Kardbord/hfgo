//go:build !integration

package dto

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestZeroShotTextClassificationRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := ZeroShotTextClassificationRequest{
		Input: "I love this product",
		Parameters: &ZeroShotTextClassificationParameters{
			CandidateLabels:    []string{"positive", "negative", "neutral"},
			HypothesisTemplate: testutils.Ptr("This example is {}."),
			MultiLabel:         testutils.Ptr(false),
		},
	}

	cloned := req.Clone()

	cloned.Parameters.CandidateLabels[0] = "great"
	*cloned.Parameters.HypothesisTemplate = "changed {}"
	*cloned.Parameters.MultiLabel = true
	cloned.Input = "changed"

	require.Equal(t, "I love this product", req.Input)
	require.Equal(t, []string{"positive", "negative", "neutral"}, req.Parameters.CandidateLabels)
	require.Equal(t, "This example is {}.", *req.Parameters.HypothesisTemplate)
	require.False(t, *req.Parameters.MultiLabel)
	require.Equal(t, "great", cloned.Parameters.CandidateLabels[0])
	require.True(t, *cloned.Parameters.MultiLabel)
}

func TestZeroShotTextClassificationBatchRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := ZeroShotTextClassificationBatchRequest{
		Inputs: []string{"text1", "text2"},
		Parameters: &ZeroShotTextClassificationParameters{
			CandidateLabels: []string{"positive", "negative"},
			MultiLabel:      testutils.Ptr(false),
		},
	}

	cloned := req.Clone()

	cloned.Inputs[0] = "changed"
	cloned.Parameters.CandidateLabels[0] = "great"

	require.Equal(t, []string{"text1", "text2"}, req.Inputs)
	require.Equal(t, []string{"positive", "negative"}, req.Parameters.CandidateLabels)
	require.Equal(t, "changed", cloned.Inputs[0])
	require.Equal(t, "great", cloned.Parameters.CandidateLabels[0])
}

func TestZeroShotTextClassificationParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &ZeroShotTextClassificationParameters{
		CandidateLabels:    []string{"positive", "negative"},
		HypothesisTemplate: testutils.Ptr("This example is {}."),
		MultiLabel:         testutils.Ptr(true),
	}

	cloned := params.Clone()

	cloned.CandidateLabels[0] = "great"
	*cloned.HypothesisTemplate = "changed {}"
	*cloned.MultiLabel = false

	require.Equal(t, []string{"positive", "negative"}, params.CandidateLabels)
	require.Equal(t, "This example is {}.", *params.HypothesisTemplate)
	require.True(t, *params.MultiLabel)
	require.Equal(t, "great", cloned.CandidateLabels[0])
	require.False(t, *cloned.MultiLabel)
}

func TestZeroShotTextClassificationClone_Nil(t *testing.T) {
	t.Parallel()

	var r *ZeroShotTextClassificationRequest
	require.Empty(t, r.Clone())

	var b *ZeroShotTextClassificationBatchRequest
	require.Empty(t, b.Clone())

	var p *ZeroShotTextClassificationParameters
	require.Empty(t, p.Clone())
}
