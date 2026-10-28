//go:build !integration

package task_test

import (
	"net/http"
	"net/http/httptest"
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

// mp4BoxSignature returns a 12-byte MP4 box header ("ftyp" brand box), a
// stand-in for arbitrary binary video data that net/http can also sniff.
func mp4BoxSignature() []byte {
	return []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p', 'm', 'p', '4', '2'}
}

func TestGenerateVideo_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []mediaDecodingCase{
		{
			name:          "mp4 video",
			output:        mp4BoxSignature(),
			outputType:    "video/mp4",
			wantMediaType: "video/mp4",
			description:   "mp4 video is returned with its media type",
		},
		{
			name:          "webm video",
			output:        []byte{0x1A, 0x45, 0xDF, 0xA3},
			outputType:    "video/webm",
			wantMediaType: "video/webm",
			description:   "webm video is returned with its media type",
		},
		{
			name:          "video with charset parameter",
			output:        mp4BoxSignature(),
			outputType:    "video/mp4; charset=utf-8",
			wantMediaType: "video/mp4",
			description:   "media-type parameters are normalized away",
		},
		{
			name:        "non-video content type",
			output:      mp4BoxSignature(),
			outputType:  "application/json",
			wantErr:     true,
			description: "non-video response is rejected",
		},
		{
			name:          "missing content type is sniffed",
			output:        []byte{0x1A, 0x45, 0xDF, 0xA3},
			outputType:    "",
			wantMediaType: "video/webm",
			description:   "missing content type falls back to byte sniffing",
		},
	}

	runMediaDecodingCases(t, cases,
		func(tc mediaDecodingCase) (hftypes.TextToVideoResponse, error) {
			server := newMediaServer(tc)
			defer server.Close()
			client := hfgo.NewClient(
				hfopts.WithBaseURL(server.URL),
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

	var (
		outputURL      string
		gotBody        map[string]any
		gotContentType string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			// The submit carries the flat WaveSpeed body.
			gotContentType = r.Header.Get("Content-Type")
			gotBody = testutils.ReadRequestJSON(t, r)

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"code":200,"message":"success","data":{` +
				`"id":"pred_1","status":"completed","outputs":["` + outputURL + `"]}}`))

			return
		}

		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write(mp4BoxSignature())
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	client := hfgo.NewClient(
		hfopts.WithBaseURL(server.URL),
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
	require.Equal(t, mp4BoxSignature(), video.Video)
	require.Equal(t, "video/mp4", video.MediaType)

	// The submit advertises JSON and accepts any media type.
	require.Equal(t, "application/json", gotContentType)

	// WaveSpeed's body is flat: the prompt and every generation parameter
	// live at the top level.
	require.Equal(t, "a majestic lion in the savanna", gotBody["prompt"])
	require.InEpsilon(t, 7.5, gotBody["guidance_scale"], 0.001)
	require.Equal(t, []any{"blurry, low quality"}, gotBody["negative_prompt"])
	require.InEpsilon(t, 30, gotBody["num_inference_steps"], 0.001)
	require.InEpsilon(t, 24, gotBody["num_frames"], 0.001)
	require.InEpsilon(t, 42, gotBody["seed"], 0.001)
}

func TestGenerateVideo_ProviderParams(t *testing.T) {
	t.Parallel()

	var (
		outputURL string
		gotBody   map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			gotBody = testutils.ReadRequestJSON(t, r)

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"code":200,"message":"success","data":{` +
				`"id":"pred_1","status":"completed","outputs":["` + outputURL + `"]}}`))

			return
		}

		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write(mp4BoxSignature())
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	client := hfgo.NewClient(
		hfopts.WithBaseURL(server.URL),
		hfopts.WithModel("test-model"),
		hfopts.WithProvider(wavespeedTestProvider()),
	)

	_, err := client.GenerateVideo(hftypes.TextToVideoRequest{
		Input: "a majestic lion in the savanna",
		ProviderParams: map[string]any{
			"duration":              5,
			"enable_safety_checker": true,
		},
	})
	require.NoError(t, err)
	require.InEpsilon(t, 5, gotBody["duration"], 0.001)
	require.Equal(t, true, gotBody["enable_safety_checker"])
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
