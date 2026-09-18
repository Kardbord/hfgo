package hfgo

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestSpeechRecognitionRequest_Clone(t *testing.T) {
	t.Parallel()

	original := SpeechRecognitionRequest{
		Input: "base64-encoded-audio",
		Parameters: &SpeechRecognitionRequestParameters{
			ReturnTimestamps: testutils.Ptr(true),
			GenerationParameters: &SpeechRecognitionGenerationParameters{
				Temperature:   testutils.Ptr(0.7),
				TopK:          testutils.Ptr(50),
				EarlyStopping: testutils.Ptr(EarlyStoppingTrue),
			},
		},
	}

	cloned := original.Clone()

	require.Equal(t, original.Input, cloned.Input)
	require.Equal(t, original.Parameters, cloned.Parameters)

	// Mutating the clone should not affect the original.
	cloned.Input = "mutated"
	cloned.Parameters.ReturnTimestamps = testutils.Ptr(false)
	cloned.Parameters.GenerationParameters.Temperature = testutils.Ptr(1.0)
	*cloned.Parameters.GenerationParameters.EarlyStopping = EarlyStoppingFalse

	require.Equal(t, "base64-encoded-audio", original.Input)
	require.True(t, *original.Parameters.ReturnTimestamps)
	require.InDelta(t, 0.7, *original.Parameters.GenerationParameters.Temperature, 0.0)
	require.Equal(t, EarlyStoppingTrue, *original.Parameters.GenerationParameters.EarlyStopping)
}

func TestSpeechRecognitionRequest_Clone_Nil(t *testing.T) {
	t.Parallel()

	var req *SpeechRecognitionRequest
	require.Empty(t, req.Clone())
}

func TestSpeechRecognitionBatchRequest_Clone(t *testing.T) {
	t.Parallel()

	original := SpeechRecognitionBatchRequest{
		Inputs: []string{"base64-audio-1", "base64-audio-2"},
		Parameters: &SpeechRecognitionRequestParameters{
			ReturnTimestamps: testutils.Ptr(false),
		},
	}

	cloned := original.Clone()

	require.Equal(t, original.Inputs, cloned.Inputs)
	require.Equal(t, original.Parameters, cloned.Parameters)

	// Mutating the clone should not affect the original.
	cloned.Inputs[0] = "mutated"
	*cloned.Parameters.ReturnTimestamps = true

	require.Equal(t, "base64-audio-1", original.Inputs[0])
	require.False(t, *original.Parameters.ReturnTimestamps)
}

func TestSpeechRecognitionBatchRequest_Clone_Nil(t *testing.T) {
	t.Parallel()

	var req *SpeechRecognitionBatchRequest
	require.Empty(t, req.Clone())
}

func TestEarlyStopping_MarshalJSON(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		value    EarlyStopping
		expected string
	}{
		{name: "true", value: EarlyStoppingTrue, expected: "true"},
		{name: "false", value: EarlyStoppingFalse, expected: "false"},
		{name: "never", value: EarlyStoppingNever, expected: `"never"`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			b, err := tc.value.MarshalJSON()
			require.NoError(t, err)
			require.Equal(t, tc.expected, string(b))
		})
	}
}

func TestEarlyStopping_UnmarshalJSON(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		input    string
		expected EarlyStopping
	}{
		{name: "boolean true", input: "true", expected: EarlyStoppingTrue},
		{name: "boolean false", input: "false", expected: EarlyStoppingFalse},
		{name: "string never", input: `"never"`, expected: EarlyStoppingNever},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var e EarlyStopping
			err := json.Unmarshal([]byte(tc.input), &e)
			require.NoError(t, err)
			require.Equal(t, tc.expected, e)
		})
	}
}

func TestEarlyStopping_UnmarshalJSON_Invalid(t *testing.T) {
	t.Parallel()

	var e EarlyStopping
	err := json.Unmarshal([]byte(`"invalid"`), &e)
	require.Error(t, err)
}

func TestSpeechRecognition_ResponseParsing(t *testing.T) {
	t.Parallel()

	responseBody := `{"text":"Hello world","chunks":[{"text":"Hello ","timestamp":[0.0,1.0]},{"text":"world","timestamp":[1.0,2.0]}]}`

	var result SpeechRecognition
	err := json.Unmarshal([]byte(responseBody), &result)
	require.NoError(t, err)
	require.Equal(t, "Hello world", result.Text)
	require.Len(t, result.Chunks, 2)
	require.Equal(t, "Hello ", result.Chunks[0].Text)
	require.Equal(t, []float64{0.0, 1.0}, result.Chunks[0].Timestamp)
	require.Equal(t, "world", result.Chunks[1].Text)
	require.Equal(t, []float64{1.0, 2.0}, result.Chunks[1].Timestamp)
}

func TestSpeechRecognition_ResponseParsing_NoChunks(t *testing.T) {
	t.Parallel()

	responseBody := `{"text":"Hello world"}`

	var result SpeechRecognition
	err := json.Unmarshal([]byte(responseBody), &result)
	require.NoError(t, err)
	require.Equal(t, "Hello world", result.Text)
	require.Nil(t, result.Chunks)
}

func TestSpeechRecognitionService_RecognizeReader_ParameterSerialization(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		params      *SpeechRecognitionRequestParameters
		want        map[string]any
		description string
	}{
		{
			name:        "params omitted",
			params:      nil,
			description: "nil parameters are omitted from the request body",
		},
		{
			name: "return timestamps",
			params: &SpeechRecognitionRequestParameters{
				ReturnTimestamps: testutils.Ptr(true),
			},
			want:        map[string]any{"return_timestamps": true},
			description: "return_timestamps maps to its JSON key",
		},
		{
			name: "generation parameters",
			params: &SpeechRecognitionRequestParameters{
				GenerationParameters: &SpeechRecognitionGenerationParameters{
					Temperature:  testutils.Ptr(0.7),
					MaxNewTokens: testutils.Ptr(100),
				},
			},
			want: map[string]any{
				"generation_parameters": map[string]any{
					"temperature":    0.7,
					"max_new_tokens": float64(100),
				},
			},
			description: "generation_parameters maps to its JSON key",
		},
		{
			name: "early stopping never",
			params: &SpeechRecognitionRequestParameters{
				GenerationParameters: &SpeechRecognitionGenerationParameters{
					EarlyStopping: testutils.Ptr(EarlyStoppingNever),
				},
			},
			want: map[string]any{
				"generation_parameters": map[string]any{"early_stopping": "never"},
			},
			description: "EarlyStoppingNever serializes as JSON string \"never\"",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				`{"text":"Test recognition."}`,
				nil,
			)
			client := NewClient(
				WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				WithModel("test-model"),
			)

			reader := bytes.NewReader([]byte("fake-audio-data"))
			_, err := client.RecognizeSpeechReader(reader, tc.params)
			require.NoError(t, err, tc.description)

			reqBody := testutils.ReadRequestBody(t, mt)
			if tc.want == nil {
				_, ok := reqBody["parameters"]
				require.False(t, ok, tc.description)

				return
			}

			require.Equal(t, tc.want, reqBody["parameters"], tc.description)
		})
	}
}

func TestSpeechRecognitionService_RecognizeReader_ReaderError(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`{"text":"Test recognition."}`,
		nil,
	)
	client := NewClient(
		WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
		WithModel("test-model"),
	)

	reader := errorReader{}
	_, err := client.RecognizeSpeechReader(reader, nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

// errorReader is a test helper that always returns an error on Read.
type errorReader struct{}

func (e errorReader) Read(_ []byte) (int, error) {
	return 0, errors.New("read failed")
}
