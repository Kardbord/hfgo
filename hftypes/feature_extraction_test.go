//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestFeatureExtractionRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := hftypes.FeatureExtractionParameters{
		Normalize:           new(true),
		PromptName:          new("query"),
		Truncate:            new(true),
		TruncationDirection: new(hftypes.FeatureExtractionTruncationRight),
	}
	single := hftypes.FeatureExtractionRequest{Input: "hello world", Parameters: &origParams}
	batch := hftypes.FeatureExtractionBatchRequest{
		Inputs:     []string{"hello", "world"},
		Parameters: &origParams,
	}

	singleCloned := single.Clone()
	batchCloned := batch.Clone()

	*singleCloned.Parameters.Normalize = false
	*singleCloned.Parameters.PromptName = "document"
	*singleCloned.Parameters.TruncationDirection = hftypes.FeatureExtractionTruncationLeft
	batchCloned.Inputs[0] = "zzz"
	*batchCloned.Parameters.Truncate = false

	require.True(t, *origParams.Normalize)
	require.Equal(t, "query", *origParams.PromptName)
	require.Equal(t, hftypes.FeatureExtractionTruncationRight, *origParams.TruncationDirection)
	require.True(t, *origParams.Truncate)
	require.Equal(t, "hello world", single.Input)
	require.Equal(t, []string{"hello", "world"}, batch.Inputs)
}

func TestFeatureExtractionRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.FeatureExtractionRequest
	require.Empty(t, r.Clone())

	var b *hftypes.FeatureExtractionBatchRequest
	require.Empty(t, b.Clone())

	var p *hftypes.FeatureExtractionParameters
	require.Empty(t, p.Clone())
}

func TestFeatureExtractionParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.FeatureExtractionParameters{
		Normalize:           new(true),
		PromptName:          new("query"),
		Truncate:            new(true),
		TruncationDirection: new(hftypes.FeatureExtractionTruncationRight),
	}

	cloned := params.Clone()

	*cloned.Normalize = false
	*cloned.PromptName = "document"
	*cloned.Truncate = false
	*cloned.TruncationDirection = hftypes.FeatureExtractionTruncationLeft

	require.True(t, *params.Normalize)
	require.Equal(t, "query", *params.PromptName)
	require.True(t, *params.Truncate)
	require.Equal(t, hftypes.FeatureExtractionTruncationRight, *params.TruncationDirection)

	require.False(t, *cloned.Normalize)
	require.Equal(t, "document", *cloned.PromptName)
	require.False(t, *cloned.Truncate)
	require.Equal(t, hftypes.FeatureExtractionTruncationLeft, *cloned.TruncationDirection)
}

func TestFeatureExtractionPooledEmbedding_Dimension(t *testing.T) {
	t.Parallel()

	require.Equal(t, 0, hftypes.FeatureExtractionPooledEmbedding(nil).Dimension())
	require.Equal(t, 0, hftypes.FeatureExtractionPooledEmbedding{}.Dimension())
	require.Equal(t, 3, hftypes.FeatureExtractionPooledEmbedding{0.1, 0.2, 0.3}.Dimension())
}

func TestFeatureExtractionPooledEmbedding_Clone(t *testing.T) {
	t.Parallel()

	require.Nil(t, hftypes.FeatureExtractionPooledEmbedding(nil).Clone())

	original := hftypes.FeatureExtractionPooledEmbedding{0.1, 0.2}
	cloned := original.Clone()
	require.Equal(t, original, cloned)

	cloned[0] = 9.9
	require.InDelta(t, 0.1, original[0], 0.0001)
}

func TestFeatureExtractionTokenEmbeddings_Dimension(t *testing.T) {
	t.Parallel()

	require.Equal(t, 0, hftypes.FeatureExtractionTokenEmbeddings(nil).Dimension())
	require.Equal(t, 0, hftypes.FeatureExtractionTokenEmbeddings{}.Dimension())
	require.Equal(
		t,
		3,
		hftypes.FeatureExtractionTokenEmbeddings{
			{0.1, 0.2, 0.3},
			{0.4, 0.5, 0.6},
		}.Dimension(),
	)
}

func TestFeatureExtractionTokenEmbeddings_Clone(t *testing.T) {
	t.Parallel()

	require.Nil(t, hftypes.FeatureExtractionTokenEmbeddings(nil).Clone())

	original := hftypes.FeatureExtractionTokenEmbeddings{
		{0.1, 0.2},
		{0.3, 0.4},
	}
	cloned := original.Clone()
	require.Equal(t, original, cloned)

	cloned[0][0] = 9.9
	require.InDelta(t, 0.1, original[0][0], 0.0001)
}
