//go:build !integration

package dto

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestSummarizationRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := SummarizationParameters{
		CleanUpTokenizationSpaces: testutils.Ptr(false),
		Truncation:                testutils.Ptr(SummarizationTruncationLongestFirst),
		GenerateParameters:        map[string]any{"max_new_tokens": 5},
	}
	single := SummarizationRequest{Input: "a", Parameters: &origParams}
	batch := SummarizationBatchRequest{Inputs: []string{"a", "b"}, Parameters: &origParams}

	singleCloned := single.Clone()
	batchCloned := batch.Clone()

	*singleCloned.Parameters.Truncation = SummarizationTruncationOnlyFirst
	singleCloned.Parameters.GenerateParameters["max_new_tokens"] = 99
	*singleCloned.Parameters.CleanUpTokenizationSpaces = true
	batchCloned.Inputs[0] = "zzz"
	*batchCloned.Parameters.Truncation = SummarizationTruncationDoNotTruncate

	require.Equal(t, SummarizationTruncationLongestFirst, *origParams.Truncation)
	require.Equal(t, "a", single.Input)
	require.Equal(t, []string{"a", "b"}, batch.Inputs)
	require.False(t, *origParams.CleanUpTokenizationSpaces)
	require.Equal(t, 5, origParams.GenerateParameters["max_new_tokens"])
}

func TestSummarizationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *SummarizationRequest
	require.Empty(t, r.Clone())

	var b *SummarizationBatchRequest
	require.Empty(t, b.Clone())

	var p *SummarizationParameters
	require.Empty(t, p.Clone())
}
