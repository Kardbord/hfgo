//go:build !integration

package dto

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestFeatureExtractionRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := FeatureExtractionParameters{
		Normalize:           testutils.Ptr(true),
		PromptName:          testutils.Ptr("query"),
		Truncate:            testutils.Ptr(true),
		TruncationDirection: testutils.Ptr(FeatureExtractionTruncationRight),
	}
	single := FeatureExtractionRequest{Input: "hello world", Parameters: &origParams}
	batch := FeatureExtractionBatchRequest{
		Inputs:     []string{"hello", "world"},
		Parameters: &origParams,
	}

	singleCloned := single.Clone()
	batchCloned := batch.Clone()

	*singleCloned.Parameters.Normalize = false
	*singleCloned.Parameters.PromptName = "document"
	*singleCloned.Parameters.TruncationDirection = FeatureExtractionTruncationLeft
	batchCloned.Inputs[0] = "zzz"
	*batchCloned.Parameters.Truncate = false

	require.True(t, *origParams.Normalize)
	require.Equal(t, "query", *origParams.PromptName)
	require.Equal(t, FeatureExtractionTruncationRight, *origParams.TruncationDirection)
	require.True(t, *origParams.Truncate)
	require.Equal(t, "hello world", single.Input)
	require.Equal(t, []string{"hello", "world"}, batch.Inputs)
}

func TestFeatureExtractionRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *FeatureExtractionRequest
	require.Empty(t, r.Clone())

	var b *FeatureExtractionBatchRequest
	require.Empty(t, b.Clone())

	var p *FeatureExtractionParameters
	require.Empty(t, p.Clone())
}

func TestFeatureExtractionParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &FeatureExtractionParameters{
		Normalize:           testutils.Ptr(true),
		PromptName:          testutils.Ptr("query"),
		Truncate:            testutils.Ptr(true),
		TruncationDirection: testutils.Ptr(FeatureExtractionTruncationRight),
	}

	cloned := params.Clone()

	*cloned.Normalize = false
	*cloned.PromptName = "document"
	*cloned.Truncate = false
	*cloned.TruncationDirection = FeatureExtractionTruncationLeft

	require.True(t, *params.Normalize)
	require.Equal(t, "query", *params.PromptName)
	require.True(t, *params.Truncate)
	require.Equal(t, FeatureExtractionTruncationRight, *params.TruncationDirection)

	require.False(t, *cloned.Normalize)
	require.Equal(t, "document", *cloned.PromptName)
	require.False(t, *cloned.Truncate)
	require.Equal(t, FeatureExtractionTruncationLeft, *cloned.TruncationDirection)
}
