//go:build !integration

package hftypes

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestTokenClassificationRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := TokenClassificationRequest{
		Input: "My name is Sarah and I live in London.",
		Parameters: &TokenClassificationParameters{
			IgnoreLabels:        []string{"O", "MISC"},
			Stride:              testutils.Ptr(5),
			AggregationStrategy: testutils.Ptr(TokenClassificationAggregationSimple),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.Stride = 10
	cloned.Parameters.IgnoreLabels[0] = "PER"
	*cloned.Parameters.AggregationStrategy = TokenClassificationAggregationMax
	cloned.Input = "changed"

	require.Equal(t, "My name is Sarah and I live in London.", req.Input)
	require.Equal(t, 5, *req.Parameters.Stride)
	require.Equal(t, []string{"O", "MISC"}, req.Parameters.IgnoreLabels)
	require.Equal(t, TokenClassificationAggregationSimple, *req.Parameters.AggregationStrategy)
	require.Equal(t, "changed", cloned.Input)
	require.Equal(t, 10, *cloned.Parameters.Stride)
	require.Equal(t, "PER", cloned.Parameters.IgnoreLabels[0])
	require.Equal(t, TokenClassificationAggregationMax, *cloned.Parameters.AggregationStrategy)
}

func TestTokenClassificationBatchRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := TokenClassificationBatchRequest{
		Inputs: []string{
			"My name is Sarah and I live in London.",
			"I work at Google in New York.",
		},
		Parameters: &TokenClassificationParameters{
			IgnoreLabels: []string{"O"},
			Stride:       testutils.Ptr(3),
		},
	}

	cloned := req.Clone()

	cloned.Inputs[0] = "changed"
	*cloned.Parameters.Stride = 7
	cloned.Parameters.IgnoreLabels[0] = "PER"

	require.Equal(t, []string{
		"My name is Sarah and I live in London.",
		"I work at Google in New York.",
	}, req.Inputs)
	require.Equal(t, 3, *req.Parameters.Stride)
	require.Equal(t, []string{"O"}, req.Parameters.IgnoreLabels)
	require.Equal(t, "changed", cloned.Inputs[0])
	require.Equal(t, 7, *cloned.Parameters.Stride)
	require.Equal(t, "PER", cloned.Parameters.IgnoreLabels[0])
}

func TestTokenClassificationParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &TokenClassificationParameters{
		IgnoreLabels:        []string{"O", "MISC"},
		Stride:              testutils.Ptr(5),
		AggregationStrategy: testutils.Ptr(TokenClassificationAggregationFirst),
	}

	cloned := params.Clone()

	*cloned.Stride = 10
	cloned.IgnoreLabels[0] = "PER"
	*cloned.AggregationStrategy = TokenClassificationAggregationAverage

	require.Equal(t, 5, *params.Stride)
	require.Equal(t, []string{"O", "MISC"}, params.IgnoreLabels)
	require.Equal(t, TokenClassificationAggregationFirst, *params.AggregationStrategy)
	require.Equal(t, 10, *cloned.Stride)
	require.Equal(t, "PER", cloned.IgnoreLabels[0])
	require.Equal(t, TokenClassificationAggregationAverage, *cloned.AggregationStrategy)
}

func TestTokenClassificationClone_Nil(t *testing.T) {
	t.Parallel()

	var r *TokenClassificationRequest
	require.Empty(t, r.Clone())

	var b *TokenClassificationBatchRequest
	require.Empty(t, b.Clone())

	var p *TokenClassificationParameters
	require.Empty(t, p.Clone())
}
