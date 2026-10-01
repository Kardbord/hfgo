//go:build !integration

package dto

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestTableQuestionAnsweringRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := TableQuestionAnsweringRequest{
		Input: TableQuestionAnsweringInput{
			Question: "How old is Bob?",
			Table: map[string][]string{
				"Name": {"Alice", "Bob", "Carol"},
				"Age":  {"25", "30", "35"},
			},
		},
		Parameters: &TableQuestionAnsweringParameters{
			Padding:    testutils.Ptr(TableQuestionAnsweringPaddingMaxLength),
			Sequential: testutils.Ptr(false),
			Truncation: testutils.Ptr(true),
		},
	}

	cloned := req.Clone()

	cloned.Input.Question = "changed"
	cloned.Input.Table["Name"][1] = "changed"
	cloned.Input.Table["City"] = []string{"NYC"}
	*cloned.Parameters.Padding = TableQuestionAnsweringPaddingDoNotPad
	*cloned.Parameters.Sequential = true

	require.Equal(t, "How old is Bob?", req.Input.Question)
	require.Equal(t, "Bob", req.Input.Table["Name"][1])
	require.Nil(t, req.Input.Table["City"])
	require.Equal(t, TableQuestionAnsweringPaddingMaxLength, *req.Parameters.Padding)
	require.False(t, *req.Parameters.Sequential)
	require.Equal(t, "changed", cloned.Input.Question)
	require.Equal(t, "changed", cloned.Input.Table["Name"][1])
	require.Equal(t, []string{"NYC"}, cloned.Input.Table["City"])
	require.Equal(t, TableQuestionAnsweringPaddingDoNotPad, *cloned.Parameters.Padding)
	require.True(t, *cloned.Parameters.Sequential)
}

func TestTableQuestionAnsweringParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &TableQuestionAnsweringParameters{
		Padding:    testutils.Ptr(TableQuestionAnsweringPaddingLongest),
		Sequential: testutils.Ptr(true),
		Truncation: testutils.Ptr(false),
	}

	cloned := params.Clone()

	*cloned.Padding = TableQuestionAnsweringPaddingMaxLength
	*cloned.Sequential = false
	*cloned.Truncation = true

	require.Equal(t, TableQuestionAnsweringPaddingLongest, *params.Padding)
	require.True(t, *params.Sequential)
	require.False(t, *params.Truncation)
	require.Equal(t, TableQuestionAnsweringPaddingMaxLength, *cloned.Padding)
	require.False(t, *cloned.Sequential)
	require.True(t, *cloned.Truncation)
}

func TestTableQuestionAnsweringClone_Nil(t *testing.T) {
	t.Parallel()

	var r *TableQuestionAnsweringRequest
	require.Empty(t, r.Clone())

	var p *TableQuestionAnsweringParameters
	require.Empty(t, p.Clone())
}
