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

func TestZeroShotClassifyText_SingleInput(t *testing.T) {
	t.Parallel()

	const zeroShotSingleClassificationResponseBody = `[{"label":"positive","score":0.95}]`
	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		zeroShotSingleClassificationResponseBody,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	candidateLabels := []string{"positive", "negative", "neutral"}
	req := hftypes.ZeroShotTextClassificationRequest{
		Input: "This is a great product!",
		Parameters: &hftypes.ZeroShotTextClassificationParameters{
			CandidateLabels: candidateLabels,
		},
	}

	result, err := client.ZeroShotClassifyText(req)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result, 1)
	require.Equal(t, "positive", result[0].Label)
	require.InEpsilon(t, 0.95, result[0].Score, 0.001)
}

func TestZeroShotClassifyText_CandidateLabelValidation(t *testing.T) {
	t.Parallel()

	const zeroShotSingleClassificationResponseBody = `[{"label":"positive","score":0.95}]`

	cases := []struct {
		name        string
		req         hftypes.ZeroShotTextClassificationRequest
		description string
	}{
		{
			name: "no candidate labels",
			req: hftypes.ZeroShotTextClassificationRequest{
				Input: "test text",
				Parameters: &hftypes.ZeroShotTextClassificationParameters{
					CandidateLabels: []string{},
				},
			},
			description: "SDK error when candidate labels are empty",
		},
		{
			name: "no parameters",
			req: hftypes.ZeroShotTextClassificationRequest{
				Input: "test text",
			},
			description: "SDK error when parameters are missing",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				zeroShotSingleClassificationResponseBody,
				nil,
			)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("nonexistent-model"),
			)

			result, err := client.ZeroShotClassifyText(tc.req)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			require.Nil(t, result, tc.description)
			require.Nil(
				t,
				mt.LastRequest,
				"candidate label validation short-circuits before any request",
			)
		})
	}
}

func TestZeroShotClassifyText_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"label":"positive","score":0.95}]`,
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
		func(opts ...hfopts.Option) ([]hftypes.ZeroShotTextClassification, error) {
			return hfgo.NewClient(opts...).
				ZeroShotClassifyText(hftypes.ZeroShotTextClassificationRequest{
					Input: "test text",
					Parameters: &hftypes.ZeroShotTextClassificationParameters{
						CandidateLabels: []string{"positive", "negative"},
					},
				})
		},
	)
}
