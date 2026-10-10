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
