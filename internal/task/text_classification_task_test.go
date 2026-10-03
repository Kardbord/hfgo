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

func TestClassifyText_SingleInput(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[{"label":"positive","score":0.95}]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	req := hftypes.TextClassificationRequest{
		Input: "test text",
	}

	result, err := client.ClassifyText(req)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result, 1)
	require.Equal(t, "positive", result[0].Label)
	require.InEpsilon(t, 0.95, result[0].Score, 0.001)
}

func TestClassifyText_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[{"label":"positive","score":0.95}]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	topK := 2
	req := hftypes.TextClassificationRequest{
		Input: "test text",
		Parameters: &hftypes.TextClassificationParameters{
			TopK: &topK,
		},
	}

	result, err := client.ClassifyText(req)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify the request was made correctly
	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.InEpsilon(t, float64(2), params["top_k"], 0.001)
}

func TestClassifyText_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[[{"label":"positive","score":0.95}]]`,
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
		func(opts ...hfopts.Option) ([]hftypes.TextClassification, error) {
			return hfgo.NewClient(opts...).ClassifyText(hftypes.TextClassificationRequest{
				Input: "test text",
			})
		},
	)
}

func TestClassifyTextBatch_ResponseVariations(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name                string
		responseBody        string
		inputs              []string
		topK                *int
		expectedOuterLength int
		expectedInnerLength int
		expectedFirstLabel  string
		expectedFirstScore  float64
		expectedSecondLabel string
		expectedSecondScore float64
		description         string
	}{
		{
			name:                "single input",
			responseBody:        `[[{"label":"positive","score":0.95}]]`,
			inputs:              []string{"test text"},
			topK:                nil,
			expectedOuterLength: 1,
			expectedInnerLength: 1,
			expectedFirstLabel:  "positive",
			expectedFirstScore:  0.95,
			description:         "single text classification",
		},
		{
			name:                "multiple inputs with TopK unset",
			responseBody:        `[[{"label":"positive","score":0.95},{"label":"negative","score":0.87},{"label":"neutral","score":0.75}]]`,
			inputs:              []string{"text1", "text2", "text3"},
			topK:                nil,
			expectedOuterLength: 3,
			expectedInnerLength: 1,
			expectedFirstLabel:  "positive",
			expectedFirstScore:  0.95,
			expectedSecondLabel: "negative",
			expectedSecondScore: 0.87,
			description:         "multiple text classifications with TopK unset (triggers normalization)",
		},
		{
			name:                "empty response",
			responseBody:        `[[]]`,
			inputs:              []string{"test text"},
			topK:                nil,
			expectedOuterLength: 1,
			expectedInnerLength: 0,
			description:         "empty classification results",
		},
		{
			name:                "multiple classifications per input with TopK set",
			responseBody:        `[[{"label":"positive","score":0.95},{"label":"negative","score":0.05}],[{"label":"negative","score":0.87},{"label":"positive","score":0.13}]]`,
			inputs:              []string{"text1", "text2"},
			topK:                testutils.Ptr(2),
			expectedOuterLength: 2,
			expectedInnerLength: 2,
			expectedFirstLabel:  "positive",
			expectedFirstScore:  0.95,
			expectedSecondLabel: "negative",
			expectedSecondScore: 0.87,
			description:         "multiple classifications with TopK parameter (no normalization)",
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

			req := hftypes.TextClassificationBatchRequest{
				Inputs: tc.inputs,
			}
			if tc.topK != nil {
				req.Parameters = &hftypes.TextClassificationParameters{
					TopK: tc.topK,
				}
			}

			result, err := client.ClassifyTextBatch(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedOuterLength, tc.description)

			if tc.expectedInnerLength > 0 {
				require.Len(t, result[0], tc.expectedInnerLength)
				require.Equal(t, tc.expectedFirstLabel, result[0][0].Label)
				require.InEpsilon(t, tc.expectedFirstScore, result[0][0].Score, 0.001)
			}

			if tc.expectedOuterLength > 1 && tc.expectedInnerLength > 0 {
				require.Len(t, result[1], tc.expectedInnerLength)
				require.Equal(t, tc.expectedSecondLabel, result[1][0].Label)
				require.InEpsilon(t, tc.expectedSecondScore, result[1][0].Score, 0.001)
			}
		})
	}
}

func TestClassifyTextBatch_NoModel(t *testing.T) {
	t.Parallel()

	// NOTE: The API documentation indicates that this should return an JSON array of
	// hftypes.TextClassification objects, but in reality it returns an array of arrays, where
	// the outer array contains only a single entry (the inner array), and the inner array
	// contains a list of hftypes.TextClassification objects.
	const batchTextClassificationResponseBody = `[[{"label":"positive","score":0.95}]]`

	mt := testutils.NewJSONMockTransport(http.StatusOK, batchTextClassificationResponseBody, nil)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	req := hftypes.TextClassificationBatchRequest{
		Inputs: []string{"test text"},
	}

	result, err := client.ClassifyTextBatch(req)
	require.Error(t, err)
	require.Nil(t, result)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	// Verify no request was made
	require.Nil(t, mt.LastRequest)
}

func TestClassifyTextBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	// NOTE: The API documentation indicates that this should return an JSON array of
	// hftypes.TextClassification objects, but in reality it returns an array of arrays, where
	// the outer array contains only a single entry (the inner array), and the inner array
	// contains a list of hftypes.TextClassification objects.
	const batchTextClassificationResponseBody = `[[{"label":"positive","score":0.95}]]`

	mt := testutils.NewJSONMockTransport(http.StatusOK, batchTextClassificationResponseBody, nil)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	req := hftypes.TextClassificationBatchRequest{
		Inputs: []string{"test text"},
	}

	result, err := client.ClassifyTextBatch(req, hfopts.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify the correct model was used in the request
	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")
}
