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

func TestSegmentImage_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		responseBody  string
		statusCode    int
		expectedLen   int
		expectedLabel string
		expectedMask  string
		expectedScore *float64
		description   string
	}{
		{
			name:          "single segment",
			responseBody:  `[{"label":"person","mask":"bWFzazE=","score":0.95}]`,
			statusCode:    http.StatusOK,
			expectedLen:   1,
			expectedLabel: "person",
			expectedMask:  "bWFzazE=",
			expectedScore: new(0.95),
			description:   "single predicted mask",
		},
		{
			name:          "multiple segments",
			responseBody:  `[{"label":"cat","mask":"bWFzazE=","score":0.93},{"label":"dog","mask":"bWFzazI=","score":0.87}]`,
			statusCode:    http.StatusOK,
			expectedLen:   2,
			expectedLabel: "cat",
			expectedMask:  "bWFzazE=",
			expectedScore: new(0.93),
			description:   "multiple predicted masks",
		},
		{
			name:          "score omitted",
			responseBody:  `[{"label":"background","mask":"bWFzaw=="}]`,
			statusCode:    http.StatusOK,
			expectedLen:   1,
			expectedLabel: "background",
			expectedMask:  "bWFzaw==",
			expectedScore: nil,
			description:   "score is optional and may be absent",
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

			segments, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
				Input: testutils.TinyPNGBase64,
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, segments)
			require.Len(t, segments, tc.expectedLen, tc.description)

			first := segments[0]
			require.Equal(t, tc.expectedLabel, first.Label)
			require.Equal(t, tc.expectedMask, first.Mask)

			if tc.expectedScore == nil {
				require.Nil(t, first.Score)
			} else {
				require.NotNil(t, first.Score)
				require.InEpsilon(t, *tc.expectedScore, *first.Score, 0.001)
			}

			if len(segments) > 1 {
				require.Equal(t, "dog", segments[1].Label)
				require.Equal(t, "bWFzazI=", segments[1].Mask)
				require.InEpsilon(t, 0.87, *segments[1].Score, 0.001)
			}
		})
	}
}

func TestSegmentImage_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"label":"hair","mask":"bWFzaw==","score":0.99}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	segments, err := client.SegmentImage(hftypes.ImageSegmentationRequest{
		Input: testutils.TinyPNGBase64,
		Parameters: &hftypes.ImageSegmentationParameters{
			MaskThreshold:            new(0.5),
			OverlapMaskAreaThreshold: new(0.25),
			Subtask:                  hftypes.ImageSegmentationSubtaskSemantic,
			Threshold:                new(0.7),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, segments)

	// Verify the request body carries every inference parameter.
	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")

	maskThreshold, ok := params["mask_threshold"].(float64)
	require.True(t, ok, "mask_threshold should be a number")
	require.InEpsilon(t, 0.5, maskThreshold, 0.001)

	overlap, ok := params["overlap_mask_area_threshold"].(float64)
	require.True(t, ok, "overlap_mask_area_threshold should be a number")
	require.InEpsilon(t, 0.25, overlap, 0.001)

	require.Equal(t, "semantic", params["subtask"])

	threshold, ok := params["threshold"].(float64)
	require.True(t, ok, "threshold should be a number")
	require.InEpsilon(t, 0.7, threshold, 0.001)
}

func TestSegmentImage_Errors(t *testing.T) {
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
		func(opts ...hfopts.Option) ([]hftypes.ImageSegmentation, error) {
			return hfgo.NewClient(opts...).SegmentImage(hftypes.ImageSegmentationRequest{
				Input: testutils.TinyPNGBase64,
			})
		},
	)
}
