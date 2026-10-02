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

func TestExtractFeatures_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		wantLen      int
		wantFirst    float64
		wantLast     float64
		wantMid      float64
		description  string
	}{
		{
			name:         "single embedding",
			responseBody: `[0.1,0.2,0.3]`,
			wantLen:      3,
			wantFirst:    0.1,
			wantLast:     0.3,
			wantMid:      0.2,
			description:  "single embedding vector is returned",
		},
		{
			name:         "empty embedding",
			responseBody: `[]`,
			wantLen:      0,
			description:  "empty embedding passes through as empty slice",
		},
		{
			name:         "high dimensional embedding",
			responseBody: `[0.01,0.02,0.03,0.04,0.05]`,
			wantLen:      5,
			wantFirst:    0.01,
			wantLast:     0.05,
			wantMid:      0.03,
			description:  "higher dimensional embedding is preserved",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			result, err := client.FeatureExtract(hftypes.FeatureExtractionRequest{
				Input: "What is the capital of France?",
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)
			require.Len(t, result, tc.wantLen, tc.description)

			if tc.wantLen > 0 {
				require.InDelta(t, tc.wantFirst, result[0], 0.0001, tc.description)
				require.InDelta(t, tc.wantMid, result[tc.wantLen/2], 0.0001, tc.description)
				require.InDelta(t, tc.wantLast, result[tc.wantLen-1], 0.0001, tc.description)
			}
		})
	}
}

func featureExtractionParameterCases() []struct {
	name        string
	params      *hftypes.FeatureExtractionParameters
	want        map[string]any
	description string
} {
	return []struct {
		name        string
		params      *hftypes.FeatureExtractionParameters
		want        map[string]any
		description string
	}{
		{
			name:        "parameters omitted",
			description: "nil parameters are omitted from the request body",
		},
		{
			name:   "normalize true",
			params: &hftypes.FeatureExtractionParameters{Normalize: testutils.Ptr(true)},
			want:   map[string]any{"normalize": true},
		},
		{
			name:   "normalize false",
			params: &hftypes.FeatureExtractionParameters{Normalize: testutils.Ptr(false)},
			want:   map[string]any{"normalize": false},
		},
		{
			name:   "prompt_name",
			params: &hftypes.FeatureExtractionParameters{PromptName: testutils.Ptr("query")},
			want:   map[string]any{"prompt_name": "query"},
		},
		{
			name:   "truncate true",
			params: &hftypes.FeatureExtractionParameters{Truncate: testutils.Ptr(true)},
			want:   map[string]any{"truncate": true},
		},
		{
			name:   "truncate false",
			params: &hftypes.FeatureExtractionParameters{Truncate: testutils.Ptr(false)},
			want:   map[string]any{"truncate": false},
		},
		{
			name: "truncation_direction left",
			params: &hftypes.FeatureExtractionParameters{
				TruncationDirection: testutils.Ptr(hftypes.FeatureExtractionTruncationLeft),
			},
			want: map[string]any{"truncation_direction": hftypes.FeatureExtractionTruncationLeft},
		},
		{
			name: "truncation_direction right",
			params: &hftypes.FeatureExtractionParameters{
				TruncationDirection: testutils.Ptr(hftypes.FeatureExtractionTruncationRight),
			},
			want: map[string]any{"truncation_direction": hftypes.FeatureExtractionTruncationRight},
		},
		{
			name: "all parameters",
			params: &hftypes.FeatureExtractionParameters{
				Normalize:           testutils.Ptr(true),
				PromptName:          testutils.Ptr("query"),
				Truncate:            testutils.Ptr(true),
				TruncationDirection: testutils.Ptr(hftypes.FeatureExtractionTruncationRight),
			},
			want: map[string]any{
				"normalize":            true,
				"prompt_name":          "query",
				"truncate":             true,
				"truncation_direction": hftypes.FeatureExtractionTruncationRight,
			},
		},
	}
}

func TestExtractFeatures_ParameterSerialization(t *testing.T) {
	t.Parallel()

	for i := range featureExtractionParameterCases() {
		tc := featureExtractionParameterCases()[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				`[0.1,0.2,0.3]`,
				nil,
			)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.FeatureExtractionRequest{Input: "test input"}
			if tc.params != nil {
				req.Parameters = tc.params
			}

			result, err := client.FeatureExtract(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)

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

func TestExtractFeatures_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[0.1,0.2,0.3]`,
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
		func(opts ...hfopts.Option) (hftypes.FeatureExtraction, error) {
			return hfgo.NewClient(opts...).FeatureExtract(hftypes.FeatureExtractionRequest{
				Input: "What is the capital of France?",
			})
		},
	)
}

func TestExtractFeaturesBatch_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		inputs       []string
		wantOuterLen int
		wantInnerLen []int
		wantFirst    float64
		description  string
	}{
		{
			name:         "single input in batch",
			responseBody: `[[0.1,0.2,0.3]]`,
			inputs:       []string{"hello"},
			wantOuterLen: 1,
			wantInnerLen: []int{3},
			wantFirst:    0.1,
			description:  "single batched input returns one embedding",
		},
		{
			name:         "multiple inputs",
			responseBody: `[[0.1,0.2,0.3],[0.4,0.5,0.6]]`,
			inputs:       []string{"hello", "world"},
			wantOuterLen: 2,
			wantInnerLen: []int{3, 3},
			wantFirst:    0.1,
			description:  "multiple batched inputs return one embedding each",
		},
		{
			name:         "empty response",
			responseBody: `[]`,
			inputs:       []string{"hello"},
			wantOuterLen: 0,
			wantInnerLen: []int{},
			description:  "empty response passes through as empty list",
		},
		{
			name:         "empty inputs",
			responseBody: `[]`,
			inputs:       []string{},
			wantOuterLen: 0,
			wantInnerLen: []int{},
			description:  "empty inputs list returns empty result",
		},
		{
			name:         "differing inner lengths",
			responseBody: `[[0.1,0.2],[0.3,0.4,0.5]]`,
			inputs:       []string{"hello", "world"},
			wantOuterLen: 2,
			wantInnerLen: []int{2, 3},
			wantFirst:    0.1,
			description:  "per-input embeddings may differ in dimension",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			result, err := client.FeatureExtractBatch(hftypes.FeatureExtractionBatchRequest{
				Inputs: tc.inputs,
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)
			require.Len(t, result, tc.wantOuterLen, tc.description)

			for j := range result {
				require.Len(t, result[j], tc.wantInnerLen[j], tc.description)
			}
			if tc.wantOuterLen > 0 && tc.wantInnerLen[0] > 0 {
				require.InDelta(t, tc.wantFirst, result[0][0], 0.0001, tc.description)
			}
		})
	}
}

func TestExtractFeaturesBatch_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[[0.1,0.2]]`,
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
		func(opts ...hfopts.Option) ([]hftypes.FeatureExtraction, error) {
			return hfgo.NewClient(opts...).
				FeatureExtractBatch(hftypes.FeatureExtractionBatchRequest{
					Inputs: []string{"hello", "world"},
				})
		},
	)
}

func TestExtractFeaturesBatch_ParameterSerialization(t *testing.T) {
	t.Parallel()

	for i := range featureExtractionParameterCases() {
		tc := featureExtractionParameterCases()[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				`[[0.1,0.2,0.3]]`,
				nil,
			)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.FeatureExtractionBatchRequest{
				Inputs: []string{"test input"},
			}
			if tc.params != nil {
				req.Parameters = tc.params
			}

			result, err := client.FeatureExtractBatch(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)

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

func TestExtractFeaturesBatch_NoModel(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[0.1,0.2,0.3]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	result, err := client.FeatureExtractBatch(hftypes.FeatureExtractionBatchRequest{
		Inputs: []string{"hello"},
	})
	require.Error(t, err)
	require.Nil(t, result)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	// Verify no request was made
	require.Nil(t, mt.LastRequest)
}

func TestExtractFeaturesBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[0.1,0.2,0.3]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	result, err := client.FeatureExtractBatch(hftypes.FeatureExtractionBatchRequest{
		Inputs: []string{"hello"},
	}, hfopts.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")

	reqBody := testutils.ReadRequestBody(t, mt)
	inputs, ok := reqBody["inputs"].([]any)
	require.True(t, ok, "inputs should be an array for batched requests")
	require.Len(t, inputs, 1)
	require.Equal(t, "hello", inputs[0])
}
