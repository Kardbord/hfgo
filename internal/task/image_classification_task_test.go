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

func TestClassifyImage_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		responseBody  string
		statusCode    int
		expectedLen   int
		expectedLabel string
		expectedScore float64
		description   string
	}{
		{
			name:          "single label",
			responseBody:  `[{"label":"Egyptian cat","score":0.514}]`,
			statusCode:    http.StatusOK,
			expectedLen:   1,
			expectedLabel: "Egyptian cat",
			expectedScore: 0.514,
			description:   "single image classification",
		},
		{
			name:          "multiple labels",
			responseBody:  `[{"label":"Egyptian cat","score":0.514},{"label":"Tabby cat","score":0.193}]`,
			statusCode:    http.StatusOK,
			expectedLen:   2,
			expectedLabel: "Egyptian cat",
			expectedScore: 0.514,
			description:   "multiple ranked image classifications",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(tc.statusCode, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			predictions, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
				Input: testutils.TinyPNGBase64,
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, predictions)
			require.Len(t, predictions, tc.expectedLen, tc.description)

			first := predictions[0]
			require.Equal(t, tc.expectedLabel, first.Label)
			require.InEpsilon(t, tc.expectedScore, first.Score, 0.001)

			if len(predictions) > 1 {
				require.Equal(t, "Tabby cat", predictions[1].Label)
				require.InEpsilon(t, 0.193, predictions[1].Score, 0.001)
			}
		})
	}
}

func TestClassifyImage_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"label":"cat","score":0.99}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	predictions, err := client.ClassifyImage(hftypes.ImageClassificationRequest{
		Input: testutils.TinyPNGBase64,
		Parameters: &hftypes.ImageClassificationParameters{
			FunctionToApply: new(hftypes.ImageClassificationFuncSoftmax),
			TopK:            new(5),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, predictions)

	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.Equal(t, hftypes.ImageClassificationFuncSoftmax, params["function_to_apply"])
	topK, ok := params["top_k"].(float64)
	require.True(t, ok, "top_k should be a number")
	require.InEpsilon(t, 5, topK, 0.001)
}

func TestClassifyImage_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[]`,
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
		func(opts ...hfopts.Option) ([]hftypes.ImageClassification, error) {
			return hfgo.NewClient(opts...).ClassifyImage(hftypes.ImageClassificationRequest{
				Input: testutils.TinyPNGBase64,
			})
		},
	)
}
