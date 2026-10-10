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

func TestFillMask_ResponseDecoding(t *testing.T) {
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
			expectedTokStr: new("Paris"),
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
			expectedTokStr: new("Paris"),
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
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.FillMaskRequest{
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

func TestFillMask_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"sequence":"The quick brown fox jumps over the lazy dog.","score":0.95,"token":1,"token_str":"lazy"}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	topK := 5
	req := hftypes.FillMaskRequest{
		Input: "The quick brown fox jumps over the [MASK] dog.",
		Parameters: &hftypes.FillMaskParameters{
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

func TestFillMask_Errors(t *testing.T) {
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
		func(opts ...hfopts.Option) ([]hftypes.FillMaskPrediction, error) {
			return hfgo.NewClient(opts...).FillMask(hftypes.FillMaskRequest{
				Input: "The capital of France is [MASK].",
			})
		},
	)
}
