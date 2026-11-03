//go:build !integration

package hfproviders

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// TestOpenAISniffAudioType pins the FLAC special case layered on top of
// net/http's byte sniffer: a bare "fLaC" prefix is labeled audio/flac, while
// everything else comes back exactly as the sniffer reports it.
func TestOpenAISniffAudioType(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		audio []byte
		want  string
	}{
		{
			name:  "flac prefix is labeled explicitly",
			audio: []byte("fLaC\x00\x00\x00\x22"),
			want:  "audio/flac",
		},
		{
			name:  "wav falls through to the sniffer",
			audio: []byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"),
			want:  "audio/wave",
		},
		{
			name:  "unknown binary falls through to the sniffer",
			audio: []byte{0x00, 0x01, 0x02, 0x03},
			want:  "application/octet-stream",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.want, openAISniffAudioType(tc.audio))
		})
	}
}

// TestOpenAITranscriptionFields pins the canonical recognition parameters and
// provider parameters as form-field values, and that nil provider entries are
// skipped. The mapped model is applied by the caller, never here.
func TestOpenAITranscriptionFields(t *testing.T) {
	t.Parallel()

	fields, err := openAITranscriptionFields(&hftypes.SpeechRecognitionRequest{
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
		ProviderParams: map[string]any{
			"language":     "en",
			"beam_size":    5,
			"temperature":  0.25,
			"skip_me":      nil,
			"json_thing":   []int{1, 2},
			"nested_thing": map[string]any{"a": 1},
		},
	})
	require.NoError(t, err)
	require.Equal(t, map[string]string{
		"return_timestamps": "true",
		"language":          "en",
		"beam_size":         "5",
		"temperature":       "0.25",
		"json_thing":        "[1,2]",
		"nested_thing":      `{"a":1}`,
	}, fields)

	t.Run("no parameters yields only provider entries", func(t *testing.T) {
		t.Parallel()

		fields, err := openAITranscriptionFields(&hftypes.SpeechRecognitionRequest{})
		require.NoError(t, err)
		require.Empty(t, fields)
	})
}

func TestOpenAIAudioExtension(t *testing.T) {
	t.Parallel()

	cases := []struct {
		mimeType string
		want     string
	}{
		{mimeType: "audio/mpeg", want: "mp3"},
		{mimeType: "audio/mp3", want: "mp3"},
		{mimeType: "audio/mp4", want: "mp4"},
		{mimeType: "video/mp4", want: "mp4"},
		{mimeType: "audio/m4a", want: "m4a"},
		{mimeType: "audio/x-m4a", want: "m4a"},
		{mimeType: "audio/flac", want: "flac"},
		{mimeType: "audio/x-flac", want: "flac"},
		{mimeType: "audio/ogg", want: "ogg"},
		{mimeType: "application/ogg", want: "ogg"},
		{mimeType: "audio/wav", want: "wav"},
		{mimeType: "audio/wave", want: "wav"},
		{mimeType: "audio/x-wav", want: "wav"},
		{mimeType: "audio/webm", want: "webm"},
		{mimeType: "video/webm", want: "webm"},
		{mimeType: "audio/wave; boundary=x", want: "wav"},
		{mimeType: "", want: "wav"},
		{mimeType: "application/octet-stream", want: "wav"},
		{mimeType: "text/plain", want: "wav"},
	}

	for _, tc := range cases {
		t.Run(tc.mimeType, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.want, openAIAudioExtension(tc.mimeType))
		})
	}
}

func TestOpenAIFormFieldValue(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		value any
		want  string
	}{
		{name: "string passes through", value: "beam", want: "beam"},
		{name: "bool true", value: true, want: "true"},
		{name: "bool false", value: false, want: "false"},
		{name: "int", value: 5, want: "5"},
		{name: "negative int", value: -7, want: "-7"},
		{name: "int8", value: int8(-8), want: "-8"},
		{name: "int16", value: int16(-16), want: "-16"},
		{name: "int32", value: int32(-32), want: "-32"},
		{name: "int64", value: int64(-64), want: "-64"},
		{name: "uint", value: uint(7), want: "7"},
		{name: "uint8", value: uint8(8), want: "8"},
		{name: "uint16", value: uint16(16), want: "16"},
		{name: "uint32", value: uint32(32), want: "32"},
		{name: "uint64", value: uint64(64), want: "64"},
		{name: "float32", value: float32(1.5), want: "1.5"},
		{name: "float64", value: 0.25, want: "0.25"},
		{name: "json number", value: json.Number("3.14"), want: "3.14"},
		{name: "map JSON-encodes", value: map[string]any{"a": 1}, want: `{"a":1}`},
		{name: "slice JSON-encodes", value: []int{1, 2}, want: "[1,2]"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := openAIFormFieldValue(tc.value)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}

	t.Run("unmarshalable value fails", func(t *testing.T) {
		t.Parallel()

		_, err := openAIFormFieldValue(make(chan int))
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestOpenAIDecodeTranscription covers the shared decode path end to end:
// the canonical transcript, segments, and the ProviderMetadata catch-all.
func TestOpenAIDecodeTranscription(t *testing.T) {
	t.Parallel()

	t.Run("transcript, segments, and metadata", func(t *testing.T) {
		t.Parallel()

		resp, err := decodeOpenAITranscription(DecodeParams{
			Context: t.Context(),
			Body: []byte(
				`{"text":"hello","segments":[{"id":0,"start":0.5,"end":1.5,"text":"hello"}],` +
					`"language":"en","duration":1.5}`,
			),
			Headers: jsonHeaders(ctJSON),
		})
		require.NoError(t, err)
		require.Equal(t, "hello", resp.Text)
		require.Len(t, resp.Chunks, 1)
		require.Equal(t, []float64{0.5, 1.5}, resp.Chunks[0].Timestamp)
		require.Equal(t, map[string]any{"language": "en", "duration": 1.5}, resp.ProviderMetadata)
	})

	t.Run("missing text fails", func(t *testing.T) {
		t.Parallel()

		_, err := decodeOpenAITranscription(DecodeParams{
			Context: t.Context(),
			Body:    []byte(`{"segments":[]}`),
			Headers: jsonHeaders(ctJSON),
		})
		require.ErrorContains(t, err, "transcription response missing text")
	})

	t.Run("non-json content type fails", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "text/plain")

		_, err := decodeOpenAITranscription(DecodeParams{
			Context: t.Context(),
			Body:    []byte(`{"text":"hi"}`),
			Headers: headers,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("invalid JSON body fails", func(t *testing.T) {
		t.Parallel()

		_, err := decodeOpenAITranscription(DecodeParams{
			Context: t.Context(),
			Body:    []byte("not json"),
			Headers: jsonHeaders(ctJSON),
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToDecodeResponseBody)
	})
}

// TestOpenAITranscriptionMetadata_UndecodableValue exercises the metadata
// decoder's defensive error branch directly: a field value that is not valid
// JSON fails with a serialization error. The full Decode path cannot produce
// such a body (the envelope already parsed as valid JSON), so the test crafts
// the field map by hand.
func TestOpenAITranscriptionMetadata_UndecodableValue(t *testing.T) {
	t.Parallel()

	metadata, err := openAITranscriptionMetadata(map[string]json.RawMessage{
		"text":  json.RawMessage(`"hi"`),
		"usage": json.RawMessage(`{"prompt_seconds":`),
	})
	require.Nil(t, metadata)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

// openAIFailWriter fails the Nth Write call (and every later one), emulating
// a destination writer that breaks mid-exchange.
type openAIFailWriter struct {
	failOn int
	writes int
}

func (w *openAIFailWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == w.failOn {
		return 0, errors.New("injected write failure")
	}

	return len(p), nil
}

// TestOpenAIMultipartBodyTo_WriteFailures exercises the multipart builder's
// defensive write-failure branches by failing the destination writer at
// every call index of the exchange. Failing each stage in turn covers the
// CreatePart, part-write, field-write, and close error branches without
// depending on how the stdlib chunks its writes; a failure point past the
// final write leaves the body intact.
func TestOpenAIMultipartBodyTo_WriteFailures(t *testing.T) {
	t.Parallel()

	fields := map[string]string{
		"return_timestamps": "true",
		"model":             "test-model",
	}

	for failOn := 1; failOn <= 16; failOn++ {
		writer := &openAIFailWriter{failOn: failOn}
		_, err := openAIMultipartBodyTo(
			writer,
			[]byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"),
			"audio/wave",
			fields,
		)

		if writer.writes < failOn {
			require.NoError(t, err, "failure point %d is past the final write", failOn)

			continue
		}

		require.Error(t, err, "failure point %d must surface", failOn)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	}
}
