//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func wavespeedTestProvider() hfproviders.WavespeedProvider {
	mapper := hfproviders.NewStaticModelResolver(map[hfproviders.StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskTextToVideo,
			Model:    "test-model",
		}: "wavespeed-ai/minimax-h3/text-to-video-lora",
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskTextToImage,
			Model:    "test-model-image",
		}: "wavespeed-ai/flux-dev",
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskImageToImage,
			Model:    "test-model-image-to-image",
		}: "wavespeed-ai/image-to-image",
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskTextToVideo,
			Model:    "nonexistent-model",
		}: "wavespeed-ai/nonexistent-model",
		{
			Provider: "wavespeed",
			Task:     hfproviders.TaskImageToImage,
			Model:    "nonexistent-model",
		}: "wavespeed-ai/nonexistent-model",
	})

	provider, err := hfproviders.NewWavespeedProvider(mapper)
	if err != nil {
		panic(err)
	}

	return provider
}

func TestGenerateVideo_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []mediaDecodingCase{
		{
			name:          "mp4 video",
			response:      []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			contentType:   "video/mp4",
			wantMediaType: "video/mp4",
			description:   "mp4 video is returned with its media type",
		},
		{
			name:          "webm video",
			response:      []byte{0x1A, 0x45, 0xDF, 0xA3},
			contentType:   "video/webm",
			wantMediaType: "video/webm",
			description:   "webm video is returned with its media type",
		},
		{
			name:          "video with charset parameter",
			response:      []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			contentType:   "video/mp4; charset=utf-8",
			wantMediaType: "video/mp4",
			description:   "media-type parameters are normalized away",
		},
		{
			name:        "non-video content type",
			response:    []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			contentType: "application/json",
			wantErr:     true,
			description: "non-video response is rejected",
		},
		{
			name:        "missing content type",
			response:    []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			contentType: "",
			wantErr:     true,
			description: "missing content type is rejected",
		},
	}

	runMediaDecodingCases(t, cases,
		func(tc mediaDecodingCase) (hftypes.TextToVideoResponse, error) {
			mt := newMediaTransport(tc.response, tc.contentType)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
				hfopts.WithProvider(wavespeedTestProvider()),
			)

			return client.GenerateVideo(hftypes.TextToVideoRequest{
				Input: "a serene mountain landscape at sunset",
			})
		},
		func(r hftypes.TextToVideoResponse) []byte { return r.Video },
		func(r hftypes.TextToVideoResponse) string { return r.MediaType },
	)
}

func TestGenerateVideo_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "video-bytes", nil)
	mt.Response.Header.Set("Content-Type", "video/mp4")
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(wavespeedTestProvider()),
	)

	video, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: "a majestic lion in the savanna",
		Parameters: &hftypes.TextToVideoParameters{
			GuidanceScale:     new(7.5),
			NegativePrompt:    new([]string{"blurry, low quality"}),
			NumInferenceSteps: new(30),
			NumFrames:         new(24),
			Seed:              new(int64(42)),
		},
	})
	require.NoError(t, err)
	require.Equal(t, []byte("video-bytes"), video.Video)
	require.Equal(t, "video/mp4", video.MediaType)

	// The request advertises JSON and accepts any media type.
	require.Equal(t, "application/json", mt.LastRequest.Header.Get("Content-Type"))
	require.Equal(t, "*/*", mt.LastRequest.Header.Get("Accept"))

	reqBody := testutils.ReadRequestBody(t, mt)
	require.Equal(t, "a majestic lion in the savanna", reqBody["inputs"])

	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.InEpsilon(t, 7.5, params["guidance_scale"], 0.001)
	require.Equal(t, []any{"blurry, low quality"}, params["negative_prompt"])
	require.InEpsilon(t, 30, params["num_inference_steps"], 0.001)
	require.InEpsilon(t, 24, params["num_frames"], 0.001)
	require.InEpsilon(t, 42, params["seed"], 0.001)
}

func TestGenerateVideo_Errors(t *testing.T) {
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
		func(opts ...hfopts.Option) (hftypes.TextToVideoResponse, error) {
			return hfgo.NewClient(append(opts, hfopts.WithProvider(wavespeedTestProvider()))...).
				GenerateVideo(hftypes.TextToVideoRequest{
					Input: "a cat",
				})
		},
	)
}
