//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestTokenClassificationRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.TokenClassificationRequest{
		Input: "My name is Sarah and I live in London.",
		Parameters: &hftypes.TokenClassificationParameters{
			IgnoreLabels:        []string{"O", "MISC"},
			Stride:              new(5),
			AggregationStrategy: new(hftypes.TokenClassificationAggregationSimple),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.Stride = 10
	cloned.Parameters.IgnoreLabels[0] = "PER"
	*cloned.Parameters.AggregationStrategy = hftypes.TokenClassificationAggregationMax
	cloned.Input = "changed"

	require.Equal(t, "My name is Sarah and I live in London.", req.Input)
	require.Equal(t, 5, *req.Parameters.Stride)
	require.Equal(t, []string{"O", "MISC"}, req.Parameters.IgnoreLabels)
	require.Equal(t, hftypes.TokenClassificationAggregationSimple,
		*req.Parameters.AggregationStrategy)
	require.Equal(t, "changed", cloned.Input)
	require.Equal(t, 10, *cloned.Parameters.Stride)
	require.Equal(t, "PER", cloned.Parameters.IgnoreLabels[0])
	require.Equal(t, hftypes.TokenClassificationAggregationMax,
		*cloned.Parameters.AggregationStrategy)
}

func TestTokenClassificationParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.TokenClassificationParameters{
		IgnoreLabels:        []string{"O", "MISC"},
		Stride:              new(5),
		AggregationStrategy: new(hftypes.TokenClassificationAggregationFirst),
	}

	cloned := params.Clone()

	*cloned.Stride = 10
	cloned.IgnoreLabels[0] = "PER"
	*cloned.AggregationStrategy = hftypes.TokenClassificationAggregationAverage

	require.Equal(t, 5, *params.Stride)
	require.Equal(t, []string{"O", "MISC"}, params.IgnoreLabels)
	require.Equal(t, hftypes.TokenClassificationAggregationFirst, *params.AggregationStrategy)
	require.Equal(t, 10, *cloned.Stride)
	require.Equal(t, "PER", cloned.IgnoreLabels[0])
	require.Equal(t, hftypes.TokenClassificationAggregationAverage, *cloned.AggregationStrategy)
}

func TestTokenClassificationClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TokenClassificationRequest
	require.Empty(t, r.Clone())

	var p *hftypes.TokenClassificationParameters
	require.Empty(t, p.Clone())
}
