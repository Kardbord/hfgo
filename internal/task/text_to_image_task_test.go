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

// pngSignature returns the 8-byte PNG file signature, a stand-in for
// arbitrary binary image data.
func pngSignature() []byte {
	return []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}
}

func TestGenerateImage_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		response      []byte
		contentType   string
		wantMediaType string
		wantErr       bool
		description   string
	}{
		{
			name:          "png image",
			response:      pngSignature(),
			contentType:   "image/png",
			wantMediaType: "image/png",
			description:   "png image is returned with its media type",
		},
		{
			name:          "jpeg image",
			response:      []byte{0xFF, 0xD8, 0xFF, 0xE0},
			contentType:   "image/jpeg",
			wantMediaType: "image/jpeg",
			description:   "jpeg image is returned with its media type",
		},
		{
			name:          "image with charset parameter",
			response:      pngSignature(),
			contentType:   "image/png; charset=utf-8",
			wantMediaType: "image/png",
			description:   "media-type parameters are normalized away",
		},
		{
			name:        "non-image content type",
			response:    pngSignature(),
			contentType: "application/json",
			wantErr:     true,
			description: "non-image response is rejected",
		},
		{
			name:        "missing content type",
			response:    pngSignature(),
			contentType: "",
			wantErr:     true,
			description: "missing content type is rejected",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewMockTransport(http.StatusOK, string(tc.response), nil)
			if tc.contentType != "" {
				mt.Response.Header.Set("Content-Type", tc.contentType)
			}
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			image, err := client.GenerateImage(hftypes.TextToImageRequest{
				Input: "a serene mountain landscape at sunset",
			})

			if tc.wantErr {
				require.Error(t, err, tc.description)
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
				require.Equal(t, hftypes.TextToImageResponse{}, image, tc.description)

				return
			}

			require.NoError(t, err, tc.description)
			require.Equal(t, tc.response, image.Image, tc.description)
			require.Equal(t, tc.wantMediaType, image.MediaType, tc.description)
		})
	}
}

func TestGenerateImage_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, string(pngSignature()), nil)
	mt.Response.Header.Set("Content-Type", "image/png")
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	image, err := client.GenerateImage(hftypes.TextToImageRequest{
		Input: "a majestic lion in the savanna",
		Parameters: &hftypes.TextToImageParameters{
			GuidanceScale:     new(7.5),
			NegativePrompt:    new("blurry, low quality"),
			NumInferenceSteps: new(30),
			Width:             new(512),
			Height:            new(512),
			Scheduler:         new("DDIM"),
			Seed:              new(int64(42)),
		},
	})
	require.NoError(t, err)
	require.Equal(t, pngSignature(), image.Image)
	require.Equal(t, "image/png", image.MediaType)

	// The request advertises JSON and accepts any image media type.
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Content-Type"))
	require.Equal(t, "image/*", mt.LastRequest.Header.Get("Accept"))

	reqBody := testutils.ReadRequestBody(t, mt)
	require.Equal(t, "a majestic lion in the savanna", reqBody["inputs"])

	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.InEpsilon(t, 7.5, params["guidance_scale"], 0.001)
	require.Equal(t, "blurry, low quality", params["negative_prompt"])
	require.InEpsilon(t, 30, params["num_inference_steps"], 0.001)
	require.InEpsilon(t, 512, params["width"], 0.001)
	require.InEpsilon(t, 512, params["height"], 0.001)
	require.Equal(t, "DDIM", params["scheduler"])
	require.InEpsilon(t, 42, params["seed"], 0.001)
}

func TestGenerateImage_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: "",
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
		func(opts ...hfopts.Option) (hftypes.TextToImageResponse, error) {
			return hfgo.NewClient(opts...).GenerateImage(hftypes.TextToImageRequest{
				Input: "a cat",
			})
		},
	)
}
