//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestFillMaskRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.FillMaskRequest{
		Input: "The capital of France is [MASK].",
		Parameters: &hftypes.FillMaskParameters{
			TopK:    new(3),
			Targets: []string{"Paris", "Lyon"},
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.TopK = 5
	cloned.Parameters.Targets[0] = "Marseille"
	cloned.Input = "changed"

	require.Equal(t, "The capital of France is [MASK].", req.Input)
	require.Equal(t, 3, *req.Parameters.TopK)
	require.Equal(t, []string{"Paris", "Lyon"}, req.Parameters.Targets)
	require.Equal(t, 5, *cloned.Parameters.TopK)
	require.Equal(t, "Marseille", cloned.Parameters.Targets[0])
}

func TestFillMaskBatchRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.FillMaskBatchRequest{
		Inputs: []string{"I [MASK] my dog.", "She is [MASK]."},
		Parameters: &hftypes.FillMaskParameters{
			TopK: new(2),
		},
	}

	cloned := req.Clone()

	cloned.Inputs[0] = "changed"
	*cloned.Parameters.TopK = 4

	require.Equal(t, []string{"I [MASK] my dog.", "She is [MASK]."}, req.Inputs)
	require.Equal(t, 2, *req.Parameters.TopK)
	require.Equal(t, "changed", cloned.Inputs[0])
	require.Equal(t, 4, *cloned.Parameters.TopK)
}

func TestFillMaskParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.FillMaskParameters{
		TopK:    new(3),
		Targets: []string{"Paris", "Lyon"},
	}

	cloned := params.Clone()

	*cloned.TopK = 7
	cloned.Targets[0] = "Marseille"

	require.Equal(t, 3, *params.TopK)
	require.Equal(t, []string{"Paris", "Lyon"}, params.Targets)
	require.Equal(t, 7, *cloned.TopK)
	require.Equal(t, "Marseille", cloned.Targets[0])
}

func TestFillMaskClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.FillMaskRequest
	require.Empty(t, r.Clone())

	var b *hftypes.FillMaskBatchRequest
	require.Empty(t, b.Clone())

	var p *hftypes.FillMaskParameters
	require.Empty(t, p.Clone())
}
