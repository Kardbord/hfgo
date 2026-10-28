//go:build !integration

package task_test

import (
	"net/http"
	"net/http/httptest"
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
			output:        []byte{0x89, 'P', 'N', 'G'},
			outputType:    "image/png",
			wantMediaType: "image/png",
			description:   "png image is returned with its media type",
		},
		{
			name:          "jpeg image",
			output:        []byte{0xFF, 0xD8, 0xFF, 0xE0},
			outputType:    "image/jpeg",
			wantMediaType: "image/jpeg",
			description:   "jpeg image is returned with its media type",
		},
		{
			name:          "image with charset parameter",
			output:        pngSignature(),
			outputType:    "image/png; charset=utf-8",
			wantMediaType: "image/png",
			description:   "media-type parameters are normalized away",
		},
		{
			name:        "non-image content type",
			output:      pngSignature(),
			outputType:  "application/json",
			wantErr:     true,
			description: "non-image response is rejected",
		},
		{
			name:          "missing content type is sniffed",
			output:        pngSignature(),
			outputType:    "",
			wantMediaType: "image/png",
			description:   "missing content type falls back to byte sniffing",
		},
	}

	runMediaDecodingCases(t, cases,
		func(tc mediaDecodingCase) (hftypes.ImageToImageResponse, error) {
			server := newMediaServer(tc)
			defer server.Close()
			client := hfgo.NewClient(
				hfopts.WithBaseURL(server.URL),
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

	const inputImage = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

	var (
		outputURL       string
		gotBody         map[string]any
		gotContentType  string
		gotAcceptHeader string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			// The submit carries the flat WaveSpeed body.
			gotContentType = r.Header.Get("Content-Type")
			gotAcceptHeader = r.Header.Get("Accept")
			gotBody = testutils.ReadRequestJSON(t, r)

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"code":200,"message":"success","data":{` +
				`"id":"pred_1","status":"completed","outputs":["` + outputURL + `"]}}`))

			return
		}

		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(pngSignature())
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	client := hfgo.NewClient(
		hfopts.WithBaseURL(server.URL),
		hfopts.WithModel("test-model-image-to-image"),
		hfopts.WithProvider(wavespeedTestProvider()),
	)

	image, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: inputImage,
		Parameters: &hftypes.ImageToImageParameters{
			Prompt:            new("Make the image blue"),
			NegativePrompt:    new("blurry"),
			GuidanceScale:     new(7.5),
			NumInferenceSteps: new(30),
		},
	})
	require.NoError(t, err)
	require.Equal(t, pngSignature(), image.Image)
	require.Equal(t, "image/png", image.MediaType)

	// The submit advertises JSON and accepts any media type.
	require.Equal(t, "application/json", gotContentType)
	require.Equal(t, "*/*", gotAcceptHeader)

	// WaveSpeed's body is flat: the source image rides on "image" (raw
	// base64 wrapped as a data URI) and the parameters live at the top
	// level.
	require.Equal(
		t,
		"data:image/jpeg;base64,"+inputImage,
		gotBody["image"],
	)
	require.Equal(t, "Make the image blue", gotBody["prompt"])
	require.Equal(t, "blurry", gotBody["negative_prompt"])
	require.InEpsilon(t, 7.5, gotBody["guidance_scale"], 0.001)
	require.InEpsilon(t, 30, gotBody["num_inference_steps"], 0.001)
}

func TestGenerateImageToImage_ProviderParams(t *testing.T) {
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

		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(pngSignature())
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	client := hfgo.NewClient(
		hfopts.WithBaseURL(server.URL),
		hfopts.WithModel("test-model-image-to-image"),
		hfopts.WithProvider(wavespeedTestProvider()),
	)

	_, err := client.GenerateImageToImage(hftypes.ImageToImageRequest{
		Input: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
		ProviderParams: map[string]any{
			"enable_base64_output": true,
		},
	})
	require.NoError(t, err)
	require.Equal(t, true, gotBody["enable_base64_output"])
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
