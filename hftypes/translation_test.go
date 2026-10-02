//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestTranslationRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := hftypes.TranslationParameters{
		CleanUpTokenizationSpaces: testutils.Ptr(false),
		SrcLang:                   testutils.Ptr("en"),
		TgtLang:                   testutils.Ptr("fr"),
		Truncation:                testutils.Ptr(hftypes.TranslationTruncationLongestFirst),
		GenerateParameters:        map[string]any{"max_new_tokens": 5},
	}
	single := hftypes.TranslationRequest{Input: "a", Parameters: &origParams}
	batch := hftypes.TranslationBatchRequest{Inputs: []string{"a", "b"}, Parameters: &origParams}

	singleCloned := single.Clone()
	batchCloned := batch.Clone()

	*singleCloned.Parameters.Truncation = hftypes.TranslationTruncationOnlyFirst
	*singleCloned.Parameters.SrcLang = "de"
	*singleCloned.Parameters.TgtLang = "es"
	singleCloned.Parameters.GenerateParameters["max_new_tokens"] = 99
	*singleCloned.Parameters.CleanUpTokenizationSpaces = true
	batchCloned.Inputs[0] = "zzz"
	*batchCloned.Parameters.Truncation = hftypes.TranslationTruncationDoNotTruncate

	require.Equal(t, hftypes.TranslationTruncationLongestFirst, *origParams.Truncation)
	require.Equal(t, "a", single.Input)
	require.Equal(t, []string{"a", "b"}, batch.Inputs)
	require.False(t, *origParams.CleanUpTokenizationSpaces)
	require.Equal(t, 5, origParams.GenerateParameters["max_new_tokens"])
}

func TestTranslationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TranslationRequest
	require.Empty(t, r.Clone())

	var b *hftypes.TranslationBatchRequest
	require.Empty(t, b.Clone())

	var p *hftypes.TranslationParameters
	require.Empty(t, p.Clone())
}
