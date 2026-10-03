//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestTableQuestionAnsweringRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.TableQuestionAnsweringRequest{
		Input: hftypes.TableQuestionAnsweringInput{
			Question: "How old is Bob?",
			Table: map[string][]string{
				"Name": {"Alice", "Bob", "Carol"},
				"Age":  {"25", "30", "35"},
			},
		},
		Parameters: &hftypes.TableQuestionAnsweringParameters{
			Padding:    testutils.Ptr(hftypes.TableQuestionAnsweringPaddingMaxLength),
			Sequential: testutils.Ptr(false),
			Truncation: testutils.Ptr(true),
		},
	}

	cloned := req.Clone()

	cloned.Input.Question = "changed"
	cloned.Input.Table["Name"][1] = "changed"
	cloned.Input.Table["City"] = []string{"NYC"}
	*cloned.Parameters.Padding = hftypes.TableQuestionAnsweringPaddingDoNotPad
	*cloned.Parameters.Sequential = true

	require.Equal(t, "How old is Bob?", req.Input.Question)
	require.Equal(t, "Bob", req.Input.Table["Name"][1])
	require.Nil(t, req.Input.Table["City"])
	require.Equal(t, hftypes.TableQuestionAnsweringPaddingMaxLength, *req.Parameters.Padding)
	require.False(t, *req.Parameters.Sequential)
	require.Equal(t, "changed", cloned.Input.Question)
	require.Equal(t, "changed", cloned.Input.Table["Name"][1])
	require.Equal(t, []string{"NYC"}, cloned.Input.Table["City"])
	require.Equal(t, hftypes.TableQuestionAnsweringPaddingDoNotPad, *cloned.Parameters.Padding)
	require.True(t, *cloned.Parameters.Sequential)
}

func TestTableQuestionAnsweringParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.TableQuestionAnsweringParameters{
		Padding:    testutils.Ptr(hftypes.TableQuestionAnsweringPaddingLongest),
		Sequential: testutils.Ptr(true),
		Truncation: testutils.Ptr(false),
	}

	cloned := params.Clone()

	*cloned.Padding = hftypes.TableQuestionAnsweringPaddingMaxLength
	*cloned.Sequential = false
	*cloned.Truncation = true

	require.Equal(t, hftypes.TableQuestionAnsweringPaddingLongest, *params.Padding)
	require.True(t, *params.Sequential)
	require.False(t, *params.Truncation)
	require.Equal(t, hftypes.TableQuestionAnsweringPaddingMaxLength, *cloned.Padding)
	require.False(t, *cloned.Sequential)
	require.True(t, *cloned.Truncation)
}

func TestTableQuestionAnsweringClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.TableQuestionAnsweringRequest
	require.Empty(t, r.Clone())

	var p *hftypes.TableQuestionAnsweringParameters
	require.Empty(t, p.Clone())
}
