//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestAnswerQuestion_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		topK         *int
		expectedLen  int
		wantAnswer   string
		wantScore    float64
		wantStart    int
		wantEnd      int
		description  string
	}{
		{
			name:         "bare object response (top_k unset)",
			responseBody: `{"answer":"Paris","score":0.95,"start":48,"end":53}`,
			topK:         nil,
			expectedLen:  1,
			wantAnswer:   "Paris",
			wantScore:    0.95,
			wantStart:    48,
			wantEnd:      53,
			description:  "API returns bare object when top_k is unset",
		},
		{
			name:         "bare object response (top_k = 1)",
			responseBody: `{"answer":"Paris","score":0.95,"start":48,"end":53}`,
			topK:         testutils.Ptr(1),
			expectedLen:  1,
			wantAnswer:   "Paris",
			wantScore:    0.95,
			wantStart:    48,
			wantEnd:      53,
			description:  "API returns bare object when top_k is 1",
		},
		{
			name:         "bare object response (top_k = 0)",
			responseBody: `{"answer":"Paris","score":0.95,"start":48,"end":53}`,
			topK:         testutils.Ptr(0),
			expectedLen:  1,
			wantAnswer:   "Paris",
			wantScore:    0.95,
			wantStart:    48,
			wantEnd:      53,
			description:  "top_k=0 falls through to bare object branch",
		},
		{
			name:         "array response (top_k > 1)",
			responseBody: `[{"answer":"Paris","score":0.95,"start":48,"end":53},{"answer":"France","score":0.03,"start":0,"end":6}]`,
			topK:         testutils.Ptr(3),
			expectedLen:  2,
			wantAnswer:   "Paris",
			wantScore:    0.95,
			wantStart:    48,
			wantEnd:      53,
			description:  "API returns array when top_k > 1",
		},
		{
			name:         "empty array response (top_k > 1)",
			responseBody: `[]`,
			topK:         testutils.Ptr(3),
			expectedLen:  0,
			description:  "no answers found",
		},
		{
			name:         "impossible answer (top_k unset)",
			responseBody: `{"answer":"","score":0.95,"start":0,"end":0}`,
			topK:         nil,
			expectedLen:  1,
			wantAnswer:   "",
			wantScore:    0.95,
			wantStart:    0,
			wantEnd:      0,
			description:  "impossible answer with empty string (bare object)",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			req := dto.QuestionAnsweringRequest{
				Input: dto.QuestionAnsweringInput{
					Question: "What is the capital of France?",
					Context:  "France is a country in Europe. Its capital is Paris.",
				},
			}
			if tc.topK != nil {
				req.Parameters = &dto.QuestionAnsweringParameters{
					TopK: tc.topK,
				}
			}

			result, err := client.AnswerQuestion(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedLen, tc.description)

			if tc.expectedLen > 0 {
				require.Equal(t, tc.wantAnswer, result[0].Answer)
				require.InEpsilon(t, tc.wantScore, result[0].Score, 0.001)
				require.Equal(t, tc.wantStart, result[0].Start)
				require.Equal(t, tc.wantEnd, result[0].End)
			}
		})
	}
}

func TestAnswerQuestion_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"answer":"Paris","score":0.95,"start":48,"end":53}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfgo.WithModel("test-model"),
	)

	topK := 5
	maxAnswerLen := 10
	handleImpossible := true
	req := dto.QuestionAnsweringRequest{
		Input: dto.QuestionAnsweringInput{
			Question: "What is the capital of France?",
			Context:  "France is a country in Europe. Its capital is Paris.",
		},
		Parameters: &dto.QuestionAnsweringParameters{
			TopK:                   &topK,
			MaxAnswerLen:           &maxAnswerLen,
			HandleImpossibleAnswer: &handleImpossible,
		},
	}

	result, err := client.AnswerQuestion(req)
	require.NoError(t, err)
	require.NotNil(t, result)

	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.InEpsilon(t, float64(5), params["top_k"], 0.001)
	require.InEpsilon(t, float64(10), params["max_answer_len"], 0.001)
	require.Equal(t, true, params["handle_impossible_answer"])
}

func TestAnswerQuestion_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"answer":"Paris","score":0.95,"start":48,"end":53}]`,
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
		func(opts ...request.Option) ([]dto.QuestionAnswering, error) {
			return hfgo.NewClient(opts...).AnswerQuestion(dto.QuestionAnsweringRequest{
				Input: dto.QuestionAnsweringInput{
					Question: "What is the capital of France?",
					Context:  "France is a country in Europe.",
				},
			})
		},
	)
}

func TestAnswerQuestion_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`{"answer":"Paris","score":0.95,"start":48,"end":53}`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	req := dto.QuestionAnsweringRequest{
		Input: dto.QuestionAnsweringInput{
			Question: "What is the capital of France?",
			Context:  "France is a country in Europe. Its capital is Paris.",
		},
	}

	result, err := client.AnswerQuestion(req, hfgo.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")

	reqBody := testutils.ReadRequestBody(t, mt)
	inputs, ok := reqBody["inputs"].(map[string]any)
	require.True(t, ok, "inputs should be an object")
	require.Equal(t, "What is the capital of France?", inputs["question"])
	require.Equal(t, "France is a country in Europe. Its capital is Paris.", inputs["context"])
}
