//go:build !integration

package hftypes_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestSpeechRecognitionRequestClone_Deep(t *testing.T) {
	t.Parallel()

	req := hftypes.SpeechRecognitionRequest{
		Input: "base64audiodata",
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	}

	cloned := req.Clone()

	*cloned.Parameters.ReturnTimestamps = false
	cloned.Input = "changed"

	require.Equal(t, "base64audiodata", req.Input)
	require.True(t, *req.Parameters.ReturnTimestamps)
	require.Equal(t, "changed", cloned.Input)
	require.False(t, *cloned.Parameters.ReturnTimestamps)
}

func TestSpeechRecognitionParametersClone_Deep(t *testing.T) {
	t.Parallel()

	params := &hftypes.SpeechRecognitionParameters{
		ReturnTimestamps: new(true),
	}

	cloned := params.Clone()

	*cloned.ReturnTimestamps = false

	require.True(t, *params.ReturnTimestamps)
	require.False(t, *cloned.ReturnTimestamps)
}

func TestSpeechRecognitionClone_Nil(t *testing.T) {
	t.Parallel()

	var r *hftypes.SpeechRecognitionRequest
	require.Empty(t, r.Clone())

	var p *hftypes.SpeechRecognitionParameters
	require.Empty(t, p.Clone())
}
