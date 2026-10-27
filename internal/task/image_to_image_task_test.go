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

func TestGenerateImageToImage_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []mediaDecodingCase{
		{
			name:          "png image",
			response:      []byte{0x89, 'P', 'N', 'G'},
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
			response:      []byte{0x89, 'P', 'N', 'G'},
			contentType:   "image/png; charset=utf-8",
			wantMediaType: "image/png",
			description:   "media-type parameters are normalized away",
		},
		{
			name:        "non-image content type",
			response:    []byte{0x89, 'P', 'N', 'G'},
			contentType: "application/json",
			wantErr:     true,
			description: "non-image response is rejected",
		},
		{
			name:        "missing content type",
			response:    []byte{0x89, 'P', 'N', 'G'},
			contentType: "",
			wantErr:     true,
			description: "missing content type is rejected",
		},
	}

	runMediaDecodingCases(t, cases,
		func(tc mediaDecodingCase) (hftypes.ImageToImageResponse, error) {
			mt := newMediaTransport(tc.response, tc.contentType)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model-image-to-image"),
				hfopts.WithProvider(wavespeedTestProvider()),
			)

			return client.GenerateImageToImage(hftypes.ImageToImageRequest{
				Input: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
			})
		},
		func(r hftypes.ImageToImageResponse) []byte { return r.Image },
		func(r hftypes.ImageToImageResponse) string { return r.MediaType },
	)
}

func TestGenerateImageToImage_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "image-bytes", nil)
	mt.Response.Header.Set("Content-Type", "image/png")
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model-image-to-image"),
		hfopts.WithProvider(wavespeedTestProvider()),
	)

	image, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
		Parameters: &hftypes.ImageToImageParameters{
			Prompt:            new("Make the image blue"),
			NegativePrompt:    new("blurry"),
			GuidanceScale:     new(7.5),
			NumInferenceSteps: new(30),
		},
	})
	require.NoError(t, err)
	require.Equal(t, []byte("image-bytes"), image.Image)
	require.Equal(t, "image/png", image.MediaType)

	// The request advertises JSON and accepts any media type.
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Content-Type"))
	require.Equal(t, "*/*", mt.LastRequest.Header.Get("Accept"))

	reqBody := testutils.ReadRequestBody(t, mt)
	require.Equal(
		t,
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
		reqBody["inputs"],
	)

	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.Equal(t, "Make the image blue", params["prompt"])
	require.Equal(t, "blurry", params["negative_prompt"])
	require.InEpsilon(t, 7.5, params["guidance_scale"], 0.001)
	require.InEpsilon(t, 30, params["num_inference_steps"], 0.001)
}

func TestGenerateImageToImage_Errors(t *testing.T) {
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
		func(opts ...hfopts.Option) (hftypes.ImageToImageResponse, error) {
			return hfgo.NewClient(append(opts, hfopts.WithProvider(wavespeedTestProvider()))...).
				GenerateImageToImage(hftypes.ImageToImageRequest{
					Input: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
				})
		},
	)
}
