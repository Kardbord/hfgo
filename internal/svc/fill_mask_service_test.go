//go:build !integration

package svc_test

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

func TestFillMaskService_FillMask_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		responseBody   string
		statusCode     int
		expectedLen    int
		expectedSeq    string
		expectedScore  float64
		expectedToken  int
		expectedTokStr *string
		expectedSeq2   string
		expectedScore2 float64
		description    string
	}{
		{
			name:           "single prediction",
			responseBody:   `[{"sequence":"The capital of France is Paris.","score":0.95,"token":1,"token_str":"Paris"}]`,
			statusCode:     http.StatusOK,
			expectedLen:    1,
			expectedSeq:    "The capital of France is Paris.",
			expectedScore:  0.95,
			expectedToken:  1,
			expectedTokStr: testutils.Ptr("Paris"),
			description:    "single mask filling prediction",
		},
		{
			name:           "null token_str",
			responseBody:   `[{"sequence":"The capital of France is Paris.","score":0.95,"token":1,"token_str":null}]`,
			statusCode:     http.StatusOK,
			expectedLen:    1,
			expectedSeq:    "The capital of France is Paris.",
			expectedScore:  0.95,
			expectedToken:  1,
			expectedTokStr: nil,
			description:    "token_str is nullable and should decode to nil",
		},
		{
			name:           "multiple ranked predictions",
			responseBody:   `[{"sequence":"The capital of France is Paris.","score":0.95,"token":1,"token_str":"Paris"},{"sequence":"The capital of France is Lyon.","score":0.03,"token":2,"token_str":"Lyon"},{"sequence":"The capital of France is Marseille.","score":0.01,"token":3,"token_str":"Marseille"}]`,
			statusCode:     http.StatusOK,
			expectedLen:    3,
			expectedSeq:    "The capital of France is Paris.",
			expectedScore:  0.95,
			expectedToken:  1,
			expectedTokStr: testutils.Ptr("Paris"),
			expectedSeq2:   "The capital of France is Lyon.",
			expectedScore2: 0.03,
			description:    "multiple ranked candidates preserve order",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(tc.statusCode, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			req := dto.FillMaskRequest{
				Input: "The capital of France is [MASK].",
			}

			result, err := client.FillMask(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedLen, tc.description)
			require.Equal(t, tc.expectedSeq, result[0].Sequence)
			require.InEpsilon(t, tc.expectedScore, result[0].Score, 0.001)
			require.Equal(t, tc.expectedToken, result[0].Token)

			if tc.expectedTokStr != nil {
				require.NotNil(t, result[0].TokenStr)
				require.Equal(t, *tc.expectedTokStr, *result[0].TokenStr)
			} else {
				require.Nil(t, result[0].TokenStr)
			}

			if tc.expectedLen > 1 {
				require.Equal(t, tc.expectedSeq2, result[1].Sequence)
				require.InEpsilon(t, tc.expectedScore2, result[1].Score, 0.001)
			}
		})
	}
}

func TestFillMaskService_FillMask_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"sequence":"The quick brown fox jumps over the lazy dog.","score":0.95,"token":1,"token_str":"lazy"}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfgo.WithModel("test-model"),
	)

	topK := 5
	req := dto.FillMaskRequest{
		Input: "The quick brown fox jumps over the [MASK] dog.",
		Parameters: &dto.FillMaskParameters{
			TopK:    &topK,
			Targets: []string{"lazy", "smart", "fast"},
		},
	}

	result, err := client.FillMask(req)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify the request was made correctly
	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.InEpsilon(t, float64(5), params["top_k"], 0.001)

	targets, ok := params["targets"].([]any)
	require.True(t, ok, "targets should be a list")
	require.Len(t, targets, 3)
	require.Equal(t, "lazy", targets[0])
	require.Equal(t, "smart", targets[1])
	require.Equal(t, "fast", targets[2])
}

func TestFillMaskService_FillMask_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"sequence":"test","score":0.95,"token":1,"token_str":"test"}]`,
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
		func(opts ...request.Option) ([]dto.FillMaskPrediction, error) {
			return hfgo.NewClient(opts...).FillMask(dto.FillMaskRequest{
				Input: "The capital of France is [MASK].",
			})
		},
	)
}

func TestFillMaskService_FillMaskBatch_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name               string
		responseBody       string
		inputs             []string
		expectedOuterLen   int
		expectedInnerLen   []int
		expectedFirstSeq   string
		expectedFirstScore float64
		description        string
	}{
		{
			name:               "single input in batch",
			responseBody:       `[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"}]]`,
			inputs:             []string{"I [MASK] my dog everyday."},
			expectedOuterLen:   1,
			expectedInnerLen:   []int{1},
			expectedFirstSeq:   "I walk my dog everyday.",
			expectedFirstScore: 0.95,
			description:        "single batched mask filling prediction",
		},
		{
			name:         "multiple inputs",
			responseBody: `[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"}],[{"sequence":"She is a brilliant programmer.","score":0.87,"token":2,"token_str":"brilliant"}]]`,
			inputs: []string{
				"I [MASK] my dog everyday.",
				"She is a [MASK] programmer.",
			},
			expectedOuterLen:   2,
			expectedInnerLen:   []int{1, 1},
			expectedFirstSeq:   "I walk my dog everyday.",
			expectedFirstScore: 0.95,
			description:        "multiple batched mask filling predictions",
		},
		{
			name:         "multiple predictions per input",
			responseBody: `[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"},{"sequence":"I pet my dog everyday.","score":0.03,"token":2,"token_str":"pet"}],[{"sequence":"She is a brilliant programmer.","score":0.87,"token":3,"token_str":"brilliant"},{"sequence":"She is a talented programmer.","score":0.04,"token":4,"token_str":"talented"}]]`,
			inputs: []string{
				"I [MASK] my dog everyday.",
				"She is a [MASK] programmer.",
			},
			expectedOuterLen:   2,
			expectedInnerLen:   []int{2, 2},
			expectedFirstSeq:   "I walk my dog everyday.",
			expectedFirstScore: 0.95,
			description:        "each batched input returns a ranked list",
		},
		{
			name:         "differing inner lengths",
			responseBody: `[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"},{"sequence":"I pet my dog everyday.","score":0.03,"token":2,"token_str":"pet"}],[{"sequence":"She is a brilliant programmer.","score":0.87,"token":3,"token_str":"brilliant"}]]`,
			inputs: []string{
				"I [MASK] my dog everyday.",
				"She is a [MASK] programmer.",
			},
			expectedOuterLen:   2,
			expectedInnerLen:   []int{2, 1},
			expectedFirstSeq:   "I walk my dog everyday.",
			expectedFirstScore: 0.95,
			description:        "per-input results are not assumed uniform",
		},
		{
			name:               "empty inner result",
			responseBody:       `[[]]`,
			inputs:             []string{"[MASK]"},
			expectedOuterLen:   1,
			expectedInnerLen:   []int{0},
			expectedFirstSeq:   "",
			expectedFirstScore: 0,
			description:        "input with zero predictions",
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

			req := dto.FillMaskBatchRequest{
				Inputs: tc.inputs,
			}

			result, err := client.FillMaskBatch(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedOuterLen, tc.description)

			for j := range result {
				require.Len(t, result[j], tc.expectedInnerLen[j], tc.description)
			}
			if len(result) > 0 && len(result[0]) > 0 {
				require.Equal(t, tc.expectedFirstSeq, result[0][0].Sequence)
				require.InEpsilon(t, tc.expectedFirstScore, result[0][0].Score, 0.001)
			}
		})
	}
}

func TestFillMaskService_FillMaskBatch_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[[]]`,
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
		func(opts ...request.Option) ([][]dto.FillMaskPrediction, error) {
			return hfgo.NewClient(opts...).FillMaskBatch(dto.FillMaskBatchRequest{
				Inputs: []string{"I [MASK] my dog everyday."},
			})
		},
	)
}

//nolint:dupl // batch model-from-options test is structurally similar across domains
func TestFillMaskService_FillMaskBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"}]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	req := dto.FillMaskBatchRequest{
		Inputs: []string{"I [MASK] my dog everyday."},
	}

	result, err := client.FillMaskBatch(req, hfgo.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify the correct model was used in the request
	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")

	// Verify the batch request body marshals "inputs" as a JSON array
	reqBody := testutils.ReadRequestBody(t, mt)
	inputs, ok := reqBody["inputs"].([]any)
	require.True(t, ok, "inputs should be an array for batched requests")
	require.Len(t, inputs, 1)
	require.Equal(t, "I [MASK] my dog everyday.", inputs[0])
}
