//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const testAudioBase64 = "AAAA"

func TestRecognizeSpeech_RawBody(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		wantText     string
		wantChunks   *[]hftypes.SpeechRecognitionChunk
		description  string
	}{
		{
			name:         "text only",
			responseBody: `{"text":"hello world"}`,
			wantText:     "hello world",
			description:  "basic transcription without timestamps",
		},
		{
			name:         "with chunks and timestamps",
			responseBody: `{"text":"hello world","chunks":[{"text":"hello","timestamp":[0.0,0.5]},{"text":" world","timestamp":[0.5,1.0]}]}`,
			wantText:     "hello world",
			wantChunks: &[]hftypes.SpeechRecognitionChunk{
				{Text: "hello", Timestamp: []float64{0.0, 0.5}},
				{Text: " world", Timestamp: []float64{0.5, 1.0}},
			},
			description: "transcription with timestamps",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
				Input: testAudioBase64,
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, resp, tc.description)
			require.Equal(t, tc.wantText, resp.Text)

			if tc.wantChunks == nil {
				require.Nil(t, resp.Chunks)
			} else {
				require.NotNil(t, resp.Chunks)
				require.Len(t, *resp.Chunks, len(*tc.wantChunks))
			}

			// Verify Content-Type was sniffed from decoded bytes
			ct := mt.LastRequest.Header.Get("Content-Type")
			require.NotEmpty(t, ct, "raw audio body should have sniffed Content-Type")
		})
	}
}

func TestRecognizeSpeech_JSONBodyWithParams(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`{"text":"hello world"}`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	resp, err := client.RecognizeSpeech(hftypes.SpeechRecognitionRequest{
		Input: testAudioBase64,
		Parameters: &hftypes.SpeechRecognitionParameters{
			ReturnTimestamps: new(true),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Equal(t, "hello world", resp.Text)

	// Verify JSON body was sent with parameters
	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, testAudioBase64, payload["inputs"])
	params, ok := payload["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	rt, ok := params["return_timestamps"].(bool)
	require.True(t, ok, "return_timestamps should be a boolean")
	require.True(t, rt)
}

func TestRecognizeSpeech_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `{"text":""}`,
				want:         testutils.WantErrSDK,
				sdkErrKind:   hferrors.SDKErrorKindConfiguration,
				description:  "SDK error when model is missing",
			},
			{
				name:         "API error on 404",
				withModel:    true,
				statusCode:   http.StatusNotFound,
				responseBody: `{"error":"Model not found"}`,
				want:         testutils.WantErrAPI,
				description:  "API error for nonexistent model",
			},
			{
				name:         "API error on 503",
				withModel:    true,
				statusCode:   http.StatusServiceUnavailable,
				responseBody: `{"error":"Model loading"}`,
				want:         testutils.WantErrAPI,
				description:  "API error for model not yet loaded",
			},
		},
		func(opts ...hfopts.Option) (hftypes.SpeechRecognitionResponse, error) {
			return hfgo.NewClient(opts...).RecognizeSpeech(hftypes.SpeechRecognitionRequest{
				Input: testAudioBase64,
			})
		},
	)
}
