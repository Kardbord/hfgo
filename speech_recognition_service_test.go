package hfgo

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestSpeechRecognitionService_Recognize_ResponseVariations(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		wantText     string
		description  string
	}{
		{
			name:         "single recognition",
			responseBody: `{"text":"Hello world"}`,
			wantText:     "Hello world",
			description:  "a single recognition is returned",
		},
		{
			name:         "recognition with chunks",
			responseBody: `{"text":"Hello world","chunks":[{"text":"Hello ","timestamp":[0.0,1.0]}]}`,
			wantText:     "Hello world",
			description:  "a recognition with timestamp chunks is returned",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := NewClient(
				WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				WithModel("test-model"),
			)

			result, err := client.RecognizeSpeech(SpeechRecognitionRequest{Input: "base64-audio"})
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)
			require.Equal(t, tc.wantText, result.Text, tc.description)
		})
	}
}

func TestSpeechRecognitionService_Recognize_ParameterSerialization(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name             string
		returnTimestamps *bool
		earlyStopping    *EarlyStopping
		temperature      *float64
		maxNewTokens     *int
		want             map[string]any
		description      string
	}{
		{
			name:        "parameters omitted",
			description: "nil parameters are omitted from the request body",
		},
		{
			name:             "return timestamps",
			returnTimestamps: testutils.Ptr(true),
			want:             map[string]any{"return_timestamps": true},
			description:      "return_timestamps maps to its JSON key",
		},
		{
			name:          "early stopping true",
			earlyStopping: testutils.Ptr(EarlyStoppingTrue),
			want: map[string]any{
				"generation_parameters": map[string]any{"early_stopping": true},
			},
			description: "EarlyStoppingTrue serializes as JSON true",
		},
		{
			name:          "early stopping never",
			earlyStopping: testutils.Ptr(EarlyStoppingNever),
			want: map[string]any{
				"generation_parameters": map[string]any{"early_stopping": "never"},
			},
			description: "EarlyStoppingNever serializes as JSON string \"never\"",
		},
		{
			name:         "generation parameters",
			temperature:  testutils.Ptr(0.7),
			maxNewTokens: testutils.Ptr(100),
			want: map[string]any{
				"generation_parameters": map[string]any{
					"temperature":    0.7,
					"max_new_tokens": float64(100),
				},
			},
			description: "generation_parameters maps to its JSON key",
		},
		{
			name:             "all parameters",
			returnTimestamps: testutils.Ptr(true),
			earlyStopping:    testutils.Ptr(EarlyStoppingNever),
			temperature:      testutils.Ptr(0.8),
			maxNewTokens:     testutils.Ptr(200),
			want: map[string]any{
				"return_timestamps": true,
				"generation_parameters": map[string]any{
					"early_stopping": "never",
					"temperature":    0.8,
					"max_new_tokens": float64(200),
				},
			},
			description: "all parameters serialize together",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
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

			req := SpeechRecognitionRequest{Input: "base64-audio"}
			hasReturnTimestamps := tc.returnTimestamps != nil
			hasGenerationParams := tc.earlyStopping != nil ||
				tc.temperature != nil || tc.maxNewTokens != nil

			if hasReturnTimestamps || hasGenerationParams {
				req.Parameters = &SpeechRecognitionRequestParameters{
					ReturnTimestamps: tc.returnTimestamps,
				}
				if hasGenerationParams {
					req.Parameters.GenerationParameters = &SpeechRecognitionGenerationParameters{
						EarlyStopping: tc.earlyStopping,
						Temperature:   tc.temperature,
						MaxNewTokens:  tc.maxNewTokens,
					}
				}
			}

			_, err := client.RecognizeSpeech(req)
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

func TestSpeechRecognitionService_Recognize_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `{"text":"Hello"}`,
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
		},
		func(opts ...Option) (SpeechRecognition, error) {
			return NewClient(opts...).RecognizeSpeech(SpeechRecognitionRequest{
				Input: "base64-audio",
			})
		},
	)
}

func TestSpeechRecognitionService_RecognizeBatch_ResponseVariations(t *testing.T) {
	t.Parallel()

	runBatchResponseVariations(
		t,
		[]batchResponseVariationCase{
			{
				name:         "single input",
				responseBody: `[{"text":"Hello"}]`,
				want:         []string{"Hello"},
				description:  "a single batched input returns one recognition",
			},
			{
				name:         "multiple inputs",
				responseBody: `[{"text":"Hello"},{"text":"World"}]`,
				want:         []string{"Hello", "World"},
				description:  "each batched input returns its own recognition",
			},
			{
				name:         "empty response",
				responseBody: `[]`,
				want:         []string{},
				description:  "empty response passes through as an empty list",
			},
		},
		func() SpeechRecognitionBatchRequest {
			return SpeechRecognitionBatchRequest{
				Inputs: []string{"base64-audio-1", "base64-audio-2"},
			}
		},
		func(c Client, req SpeechRecognitionBatchRequest) ([]SpeechRecognition, error) {
			return c.RecognizeSpeechBatch(req)
		},
		func(r SpeechRecognition) string { return r.Text },
	)
}

func TestSpeechRecognitionService_RecognizeBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"text":"Hello"}]`,
		nil,
	)
	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	result, err := client.RecognizeSpeechBatch(SpeechRecognitionBatchRequest{
		Inputs: []string{"base64-audio-1"},
	}, WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")
}
