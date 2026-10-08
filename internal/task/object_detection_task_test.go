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

func TestDetectObjects_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		responseBody  string
		statusCode    int
		expectedLen   int
		expectedLabel string
		expectedScore float64
		expectedBox   hftypes.ObjectDetectionBoundingBox
		description   string
	}{
		{
			name:          "single detection",
			responseBody:  `[{"label":"person","score":0.95,"box":{"xmin":100,"ymin":50,"xmax":200,"ymax":250}}]`,
			statusCode:    http.StatusOK,
			expectedLen:   1,
			expectedLabel: "person",
			expectedScore: 0.95,
			expectedBox: hftypes.ObjectDetectionBoundingBox{
				XMin: 100, YMin: 50, XMax: 200, YMax: 250,
			},
			description: "single object detection",
		},
		{
			name:          "multiple detections",
			responseBody:  `[{"label":"dog","score":0.93,"box":{"xmin":10,"ymin":20,"xmax":110,"ymax":120}},{"label":"cat","score":0.87,"box":{"xmin":200,"ymin":150,"xmax":300,"ymax":250}}]`,
			statusCode:    http.StatusOK,
			expectedLen:   2,
			expectedLabel: "dog",
			expectedScore: 0.93,
			expectedBox: hftypes.ObjectDetectionBoundingBox{
				XMin: 10, YMin: 20, XMax: 110, YMax: 120,
			},
			description: "multiple objects detected",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(tc.statusCode, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			// Use a minimal base64 image (1x1 transparent PNG)
			img := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAO6oxnYAAAAASUVORK5CYII="

			objects, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
				Input: img,
			})
			require.NoError(t, err, tc.description)
			require.NotNil(t, objects)
			require.Len(t, objects, tc.expectedLen, tc.description)

			first := objects[0]
			require.Equal(t, tc.expectedLabel, first.Label)
			require.InEpsilon(t, tc.expectedScore, first.Score, 0.001)
			require.Equal(t, tc.expectedBox, first.Box)

			if len(objects) > 1 {
				require.Equal(t, "cat", objects[1].Label)
				require.InEpsilon(t, 0.87, objects[1].Score, 0.001)
			}
		})
	}
}

func TestDetectObjects_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"label":"car","score":0.99,"box":{"xmin":0,"ymin":0,"xmax":10,"ymax":10}}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	threshold := 0.7
	img := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAO6oxnYAAAAASUVORK5CYII="
	objects, err := client.DetectObjects(hftypes.ObjectDetectionRequest{
		Input: img,
		Parameters: &hftypes.ObjectDetectionParameters{
			Threshold: &threshold,
		},
	})
	require.NoError(t, err)
	require.NotNil(t, objects)

	// Verify the request body contains the threshold parameter
	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	thr, ok := params["threshold"].(float64)
	require.True(t, ok, "threshold should be a number")
	require.InEpsilon(t, 0.7, thr, 0.001)
}

func TestDetectObjects_Errors(t *testing.T) {
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
		func(opts ...hfopts.Option) ([]hftypes.ObjectDetection, error) {
			img := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAO6oxnYAAAAASUVORK5CYII="

			return hfgo.NewClient(opts...).DetectObjects(hftypes.ObjectDetectionRequest{
				Input: img,
			})
		},
	)
}
