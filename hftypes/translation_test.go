//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestTranslationRequestClone_NestedIndependence(t *testing.T) {
	t.Parallel()

	origParams := hftypes.TranslationParameters{
		CleanUpTokenizationSpaces: new(false),
		SrcLang:                   new("en"),
		TgtLang:                   new("fr"),
		Truncation:                new(hftypes.TranslationTruncationLongestFirst),
		GenerateParameters:        map[string]any{"max_new_tokens": 5},
	}
	single := hftypes.TranslationRequest{Input: "a", Parameters: &origParams}

	singleCloned := single.Clone()

	*singleCloned.Parameters.Truncation = hftypes.TranslationTruncationOnlyFirst
	*singleCloned.Parameters.SrcLang = "de"
	*singleCloned.Parameters.TgtLang = "es"
	singleCloned.Parameters.GenerateParameters["max_new_tokens"] = 99
	*singleCloned.Parameters.CleanUpTokenizationSpaces = true

	require.Equal(t, hftypes.TranslationTruncationLongestFirst, *origParams.Truncation)
	require.Equal(t, "a", single.Input)
	require.False(t, *origParams.CleanUpTokenizationSpaces)
	require.Equal(t, 5, origParams.GenerateParameters["max_new_tokens"])
}

func TestTranslationRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TranslationRequest
	require.Empty(t, r.Clone())

	var p *hftypes.TranslationParameters
	require.Empty(t, p.Clone())
}
