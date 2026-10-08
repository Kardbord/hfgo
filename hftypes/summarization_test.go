//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestSummarizationRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := hftypes.SummarizationParameters{
		CleanUpTokenizationSpaces: new(false),
		Truncation:                new(hftypes.SummarizationTruncationLongestFirst),
		GenerateParameters:        map[string]any{"max_new_tokens": 5},
	}
	single := hftypes.SummarizationRequest{Input: "a", Parameters: &origParams}
	batch := hftypes.SummarizationBatchRequest{Inputs: []string{"a", "b"}, Parameters: &origParams}

	singleCloned := single.Clone()
	batchCloned := batch.Clone()

	*singleCloned.Parameters.Truncation = hftypes.SummarizationTruncationOnlyFirst
	singleCloned.Parameters.GenerateParameters["max_new_tokens"] = 99
	*singleCloned.Parameters.CleanUpTokenizationSpaces = true
	batchCloned.Inputs[0] = "zzz"
	*batchCloned.Parameters.Truncation = hftypes.SummarizationTruncationDoNotTruncate

	require.Equal(t, hftypes.SummarizationTruncationLongestFirst, *origParams.Truncation)
	require.Equal(t, "a", single.Input)
	require.Equal(t, []string{"a", "b"}, batch.Inputs)
	require.False(t, *origParams.CleanUpTokenizationSpaces)
	require.Equal(t, 5, origParams.GenerateParameters["max_new_tokens"])
}

func TestSummarizationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.SummarizationRequest
	require.Empty(t, r.Clone())

	var b *hftypes.SummarizationBatchRequest
	require.Empty(t, b.Clone())

	var p *hftypes.SummarizationParameters
	require.Empty(t, p.Clone())
}
