//go:build !integration

package hfproviders

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestWaveSpeedProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWaveSpeedProvider(mapper)
	require.NoError(t, err)

	require.Equal(t, "wavespeed", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

func TestNewWaveSpeedProvider_NilResolver(t *testing.T) {
	t.Parallel()

	p, err := NewWaveSpeedProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, WaveSpeedProvider{}, p)
}

func TestWaveSpeedProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var p WaveSpeedProvider

	_, err := p.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = p.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = p.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// wavespeedEndpointCase pairs a display name with the endpoint method it
// exercises and the WaveSpeed mapping that method must resolve to.
type wavespeedEndpointCase struct {
	name            string
	task            Task
	model           string
	providerModelID string
	endpoint        func(WaveSpeedProvider, EndpointParams) (Endpoint, error)
}

func TestWaveSpeedProvider_Endpoints(t *testing.T) {
	t.Parallel()

	cases := []wavespeedEndpointCase{
		{
			name:            "TextToVideoEndpoint",
			task:            TaskTextToVideo,
			model:           "ZhengmingYu/DMAD",
			providerModelID: "wavespeed-ai/minimax-h3/text-to-video-lora",
			endpoint:        WaveSpeedProvider.TextToVideoEndpoint,
		},
		{
			name:            "TextToImageEndpoint",
			task:            TaskTextToImage,
			model:           "black-forest-labs/FLUX.1-dev",
			providerModelID: "wavespeed-ai/flux-dev",
			endpoint:        WaveSpeedProvider.TextToImageEndpoint,
		},
		{
			name:            "ImageToImageEndpoint",
			task:            TaskImageToImage,
			model:           "black-forest-labs/FLUX.2-klein-base-4B",
			providerModelID: "bria/fibo-2.0/edit",
			endpoint:        WaveSpeedProvider.ImageToImageEndpoint,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: "wavespeed",
					Task:     tc.task,
					Model:    tc.model,
				}: {ProviderModelID: tc.providerModelID},
			})
			p, err := NewWaveSpeedProvider(mapper)
			require.NoError(t, err)

			endpoint, err := tc.endpoint(p, EndpointParams{
				Context: context.Background(),
				Model:   tc.model,
			})
			require.NoError(t, err)
			require.Equal(t, http.MethodPost, endpoint.Method)
			require.Equal(t, "wavespeed/api/v3/"+tc.providerModelID, endpoint.Path)
			require.NotNil(t, endpoint.Mapping)
			require.Equal(t, tc.providerModelID, endpoint.Mapping.ProviderModelID)
		})
	}
}

func TestWaveSpeedProvider_Endpoint_ResolutionError(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWaveSpeedProvider(mapper)
	require.NoError(t, err)

	_, err = p.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "unknown-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// requireFlatEncode asserts an encoder produced the expected flat WaveSpeed
// body with JSON content headers.
func requireFlatEncode[Req any](
	t *testing.T,
	encode func(EncodeParams[Req]) ([]byte, http.Header, error),
	req Req,
	wantJSON string,
) {
	t.Helper()

	body, headers, err := encode(EncodeParams[Req]{
		Context: context.Background(),
		Request: req,
	})
	require.NoError(t, err)
	require.JSONEq(t, wantJSON, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "*/*", headers.Get("Accept"))
}

func TestWaveSpeedProvider_TextToVideoCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := wavespeedTextToVideoCodec{}

	t.Run("flat body with parameters", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToVideoParameters{
					NumFrames: new(24),
					Seed:      new(int64(42)),
				},
			},
			`{"prompt":"a cat","num_frames":24,"seed":42}`,
		)
	})

	t.Run("nil parameters yields prompt only", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{Input: "a cat"},
			`{"prompt":"a cat"}`,
		)
	})

	t.Run("provider params merged and override typed fields", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToVideoParameters{
					NumFrames: new(24),
				},
				ProviderParams: map[string]any{
					"duration":              5,
					"enable_safety_checker": true,
					"num_frames":            48, // overrides the typed field
				},
			},
			`{"prompt":"a cat","num_frames":48,"duration":5,"enable_safety_checker":true}`,
		)
	})
}

func TestWaveSpeedProvider_TextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := wavespeedTextToImageCodec{}

	t.Run("flat body with parameters", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					Width: new(512),
				},
			},
			`{"prompt":"a cat","width":512}`,
		)
	})

	t.Run("provider params merged", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					"enable_safety_checker": true,
				},
			},
			`{"prompt":"a cat","enable_safety_checker":true}`,
		)
	})
}

func TestWaveSpeedProvider_ImageToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := wavespeedImageToImageCodec{}

	t.Run("data URI passes through", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				Parameters: &hftypes.ImageToImageParameters{
					Prompt: new("make it blue"),
				},
			},
			`{
				"image": "data:image/png;base64,abc123",
				"images": ["data:image/png;base64,abc123"],
				"prompt": "make it blue"
			}`,
		)
	})

	t.Run("http URL passes through", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "https://example.com/cat.png",
			},
			`{
				"image": "https://example.com/cat.png",
				"images": ["https://example.com/cat.png"]
			}`,
		)
	})

	t.Run("raw base64 wrapped as jpeg data URI", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{Input: "abc123"},
			`{
				"image": "data:image/jpeg;base64,abc123",
				"images": ["data:image/jpeg;base64,abc123"]
			}`,
		)
	})

	t.Run("parameters and provider params merged", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				Parameters: &hftypes.ImageToImageParameters{
					Prompt:            new("make it blue"),
					GuidanceScale:     new(7.5),
					NumInferenceSteps: new(30),
				},
				ProviderParams: map[string]any{
					"enable_base64_output": true,
				},
			},
			`{
				"image": "data:image/png;base64,abc123",
				"images": ["data:image/png;base64,abc123"],
				"prompt": "make it blue",
				"guidance_scale": 7.5,
				"num_inference_steps": 30,
				"enable_base64_output": true
			}`,
		)
	})

	t.Run("provider params override images for multi-image models", func(t *testing.T) {
		t.Parallel()

		requireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				ProviderParams: map[string]any{
					"images": []string{
						"https://example.com/one.png",
						"https://example.com/two.png",
					},
				},
			},
			`{
				"image": "data:image/png;base64,abc123",
				"images": ["https://example.com/one.png", "https://example.com/two.png"]
			}`,
		)
	})
}

// TestWaveSpeedProvider_AdapterLorasInjection exercises the LoRA payload
// injection: a LoRA-adapter mapping injects the "loras" entry naming the
// Hugging Face model ID, a caller-supplied "loras" wins, non-adapter models
// and nil resolvers leave the payload untouched, and lookup failures
// propagate.
func TestWaveSpeedProvider_AdapterLorasInjection(t *testing.T) {
	t.Parallel()

	t.Run("injected on all tasks", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "wavespeed",
				Task:     TaskTextToVideo,
				Model:    "ZhengmingYu/DMAD",
			}: {
				ProviderModelID:    "wavespeed-ai/minimax-h3/text-to-video-lora",
				Adapter:            "lora",
				AdapterWeightsPath: "minimax_h3/dmad.safetensors",
				Status:             "live",
			},
			{
				Provider: "wavespeed",
				Task:     TaskTextToImage,
				Model:    "some-lora/flux-lora",
			}: {ProviderModelID: "wavespeed-ai/flux-dev", Adapter: "lora"},
			{
				Provider: "wavespeed",
				Task:     TaskImageToImage,
				Model:    "some-lora/flux-lora",
			}: {ProviderModelID: "wavespeed-ai/flux-dev", Adapter: "lora"},
		})

		video := WaveSpeedTextToVideoMixin{Resolver: resolver}.TextToVideoCodec()
		body, _, err := video.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"prompt":"a cat","loras":[{"path":"ZhengmingYu/DMAD","scale":1}]}`,
			string(body),
		)

		image := WaveSpeedTextToImageMixin{Resolver: resolver}.TextToImageCodec()
		body, _, err = image.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"prompt":"a cat","loras":[{"path":"some-lora/flux-lora","scale":1}]}`,
			string(body),
		)

		i2i := WaveSpeedImageToImageMixin{Resolver: resolver}.ImageToImageCodec()
		body, _, err = i2i.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{Input: "abc123"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{
				"image": "data:image/jpeg;base64,abc123",
				"images": ["data:image/jpeg;base64,abc123"],
				"loras": [{"path": "some-lora/flux-lora", "scale": 1}]
			}`,
			string(body),
		)
	})

	t.Run("caller-supplied loras win", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "wavespeed",
				Task:     TaskTextToVideo,
				Model:    "ZhengmingYu/DMAD",
			}: {ProviderModelID: "wavespeed-ai/minimax-h3/text-to-video-lora", Adapter: "lora"},
		})
		codec := WaveSpeedTextToVideoMixin{Resolver: resolver}.TextToVideoCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					"loras": []map[string]any{{"path": "user/own-lora", "scale": 0.8}},
				},
			},
			Model: "ZhengmingYu/DMAD",
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"prompt":"a cat","loras":[{"path":"user/own-lora","scale":0.8}]}`,
			string(body),
		)
	})

	t.Run("non-adapter mapping injects nothing", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "wavespeed",
				Task:     TaskTextToVideo,
				Model:    "black-forest-labs/FLUX.1-dev",
			}: {ProviderModelID: "wavespeed-ai/flux-dev"},
		})
		codec := WaveSpeedTextToVideoMixin{Resolver: resolver}.TextToVideoCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "black-forest-labs/FLUX.1-dev",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"prompt":"a cat"}`, string(body))
	})

	t.Run("nil resolver injects nothing", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedTextToVideoCodec{}

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"prompt":"a cat"}`, string(body))
	})

	t.Run("lookup failure propagates", func(t *testing.T) {
		t.Parallel()

		codec := WaveSpeedTextToVideoMixin{Resolver: NewStaticModelResolver(nil)}.TextToVideoCodec()

		_, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "unknown-model",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

// TestWaveSpeedProvider_EncodePrefersEndpointMapping verifies that Encode
// reuses the mapping resolved at endpoint-construction time
// ([EncodeParams.Mapping]) instead of looking it up again: the LoRA entry is
// injected from the supplied mapping while the codec's resolver is never
// called. When no mapping is supplied, the resolver is consulted exactly
// once.
func TestWaveSpeedProvider_EncodePrefersEndpointMapping(t *testing.T) {
	t.Parallel()

	t.Run("supplied mapping used without resolver call", func(t *testing.T) {
		t.Parallel()

		// The resolver would answer with a non-adapter mapping: if Encode
		// consulted it, no loras entry would appear in the body.
		resolver := &recordingResolver{
			mapping: ModelMapping{ProviderModelID: "wavespeed-ai/flux-dev"},
		}
		codec := WaveSpeedTextToVideoMixin{Resolver: resolver}.TextToVideoCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
			Mapping: &ModelMapping{
				ProviderModelID: "wavespeed-ai/minimax-h3/text-to-video-lora",
				Adapter:         "lora",
			},
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"prompt":"a cat","loras":[{"path":"ZhengmingYu/DMAD","scale":1}]}`,
			string(body),
		)
		require.Zero(t, resolver.calls.Load())
	})

	t.Run("nil mapping falls back to resolver", func(t *testing.T) {
		t.Parallel()

		resolver := &recordingResolver{
			mapping: ModelMapping{ProviderModelID: "wavespeed-ai/flux-dev", Adapter: "lora"},
		}
		codec := WaveSpeedTextToVideoMixin{Resolver: resolver}.TextToVideoCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"prompt":"a cat","loras":[{"path":"ZhengmingYu/DMAD","scale":1}]}`,
			string(body),
		)
		require.Equal(t, int64(1), resolver.calls.Load())
	})
}

func TestWaveSpeedProvider_TextToVideoCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := wavespeedTextToVideoCodec{wavespeedAsyncCodec{
		queueTaskCodec{mediaPrefix: "video/"},
	}}

	t.Run("valid video", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "video/mp4; charset=utf-8")
		body := []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'}

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    body,
			Headers: headers,
		})
		require.NoError(t, err)
		require.Equal(t, body, resp.Video)
		require.Equal(t, "video/mp4", resp.MediaType)
	})

	t.Run("invalid content type", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"foo":"bar"}`),
			Headers: headers,
		})
		require.Error(t, err)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestWaveSpeedProvider_ImageCodecDecodes exercises the image-output codecs
// shared by the text-to-image and image-to-image mixins.
func TestWaveSpeedProvider_ImageCodecs_Decode(t *testing.T) {
	t.Parallel()

	t.Run("text-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedTextToImageCodec{wavespeedAsyncCodec{
			queueTaskCodec{mediaPrefix: "image/"},
		}}
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.TextToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})

	t.Run("image-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedImageToImageCodec{wavespeedAsyncCodec{
			queueTaskCodec{mediaPrefix: "image/"},
		}}
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.ImageToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})
}

// wavespeedSubsetProvider serves only text-to-image by embedding the
// corresponding WaveSpeed mixin with a resolver set directly, without going
// through NewWaveSpeedProvider.
type wavespeedSubsetProvider struct {
	WaveSpeedTextToImageMixin
}

func (wavespeedSubsetProvider) ProviderSuffix() string { return "" }

func (wavespeedSubsetProvider) Name() string { return "subset" }

func TestWaveSpeedMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	p := wavespeedSubsetProvider{
		WaveSpeedTextToImageMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: "wavespeed",
					Task:     TaskTextToImage,
					Model:    "black-forest-labs/FLUX.1-dev",
				}: {ProviderModelID: "wavespeed-ai/flux-dev"},
			}),
		},
	}

	prov, err := AsProvider[TextToImageProvider](p)
	require.NoError(t, err)

	endpoint, err := prov.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "black-forest-labs/FLUX.1-dev",
	})
	require.NoError(t, err)
	require.Equal(t, "wavespeed/api/v3/wavespeed-ai/flux-dev", endpoint.Path)

	_, err = AsProvider[TextToVideoProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.TextToVideoProvider",
	)

	_, err = AsProvider[ImageToImageProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.ImageToImageProvider",
	)
}

func TestWaveSpeedMixins_ZeroValueResolvers(t *testing.T) {
	t.Parallel()

	var (
		videoMixin  WaveSpeedTextToVideoMixin
		imageMixin  WaveSpeedTextToImageMixin
		image2Mixin WaveSpeedImageToImageMixin
	)

	_, err := videoMixin.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = imageMixin.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = image2Mixin.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestWaveSpeedMixins_CodecsAsync asserts every WaveSpeed mixin serves an
// AsyncCodec, so the task pipeline routes their responses through PostSubmit.
func TestWaveSpeedMixins_CodecsAsync(t *testing.T) {
	t.Parallel()

	p, err := NewWaveSpeedProvider(NewStaticModelResolver(nil))
	require.NoError(t, err)

	videoAsync, ok := any(p.TextToVideoCodec()).(AsyncCodec[hftypes.TextToVideoRequest, hftypes.TextToVideoResponse])
	require.True(t, ok, "text-to-video codec must implement AsyncCodec")
	imageAsync, ok := any(p.TextToImageCodec()).(AsyncCodec[hftypes.TextToImageRequest, hftypes.TextToImageResponse])
	require.True(t, ok, "text-to-image codec must implement AsyncCodec")
	imageToImageAsync, ok := any(p.ImageToImageCodec()).(AsyncCodec[hftypes.ImageToImageRequest, hftypes.ImageToImageResponse])
	require.True(t, ok, "image-to-image codec must implement AsyncCodec")

	require.NotNil(t, videoAsync)
	require.NotNil(t, imageAsync)
	require.NotNil(t, imageToImageAsync)
}

// fastImageCodec returns a text-to-image codec with fast polling, so tests
// exercise the exchange's branching without real delays.
func fastImageCodec() wavespeedTextToImageCodec {
	return wavespeedTextToImageCodec{wavespeedAsyncCodec{queueTaskCodec{
		mediaPrefix:     "image/",
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// fastVideoCodec mirrors fastImageCodec for the text-to-video prefix.
func fastVideoCodec() wavespeedTextToVideoCodec {
	return wavespeedTextToVideoCodec{wavespeedAsyncCodec{queueTaskCodec{
		mediaPrefix:     "video/",
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// fastImageToImageCodec mirrors fastImageCodec for image-to-image.
func fastImageToImageCodec() wavespeedImageToImageCodec {
	return wavespeedImageToImageCodec{wavespeedAsyncCodec{queueTaskCodec{
		mediaPrefix:     "image/",
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// wavespeedTaskBody builds a WaveSpeed task response body for the given data
// fields. It marshals the response through encoding/json rather than string
// concatenation, so any error message or URL is escaped correctly.
func wavespeedTaskBody(status, resultPath, taskErr string, outputs []string) string {
	task := map[string]any{
		"id":     "pred_123",
		"model":  "wavespeed-ai/flux-dev",
		"status": status,
	}
	if resultPath != "" {
		task["urls"] = map[string]any{"get": resultPath}
	}
	if taskErr != "" {
		task["error"] = taskErr
	}
	if outputs != nil {
		task["outputs"] = outputs
	}

	body, err := json.Marshal(map[string]any{
		"code":    200,
		"message": "success",
		"data":    task,
	})
	if err != nil {
		// Unreachable: every value above marshals without error.
		panic(err)
	}

	return string(body)
}

// TestWaveSpeedPostSubmit_SyncModeComplete exercises the exchange when the
// submit response already carries the completed task (sync mode): the output
// is downloaded without any polling requests.
func TestWaveSpeedPostSubmit_SyncModeComplete(t *testing.T) {
	t.Parallel()

	var (
		polls         atomic.Int64
		downloads     atomic.Int64
		gotDownloadUA string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v3/predictions/pred_123/result" {
			polls.Add(1)
		} else {
			downloads.Add(1)
			gotDownloadUA = r.Header.Get("User-Agent")
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{server.URL + "/output.png"})),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(0), polls.Load(), "a completed submit response needs no polling")
	require.Equal(t, int64(1), downloads.Load())
	require.Equal(t, "test-agent/1.0", gotDownloadUA)
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
	require.Equal(t, "wavespeed-ai/flux-dev", resp.ProviderMetadata["model"])
}

// TestWaveSpeedPostSubmit_InvalidOutputURL pins the structural credential
// guard: an output URL that is not an absolute http(s) URL is a
// configuration error, and no request is made for it.
func TestWaveSpeedPostSubmit_InvalidOutputURL(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("no request should be made for an invalid output URL")
	}))
	defer server.Close()

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{"cdn/output.png"})),
		http.Header{},
	)

	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestWaveSpeedPostSubmit_PollUntilComplete exercises the full exchange when
// the task is still running on submit: the result URL is polled (with the
// caller's bearer token) until it completes, then the output is downloaded
// unauthenticated from the provider CDN.
func TestWaveSpeedPostSubmit_PollUntilComplete(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var (
		polls           atomic.Int64
		downloads       atomic.Int64
		outputURL       string
		gotPollAuth     string
		gotPollAccept   string
		gotPollUA       string
		gotDownloadAuth string
		gotDownloadUA   string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case resultPath:
			gotPollAuth = r.Header.Get("Authorization")
			gotPollAccept = r.Header.Get("Accept")
			gotPollUA = r.Header.Get("User-Agent")

			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write([]byte(wavespeedTaskBody("processing", resultPath, "", nil)))

				return
			}
			_, _ = w.Write(
				[]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})),
			)
		case "/cdn/output.png":
			downloads.Add(1)
			gotDownloadAuth = r.Header.Get("Authorization")
			gotDownloadUA = r.Header.Get("User-Agent")

			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		default:
			t.Errorf("unexpected request path %q", r.URL.Path)
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())

	// Polls carry the caller's bearer token and JSON accept; CDN downloads
	// are unauthenticated. Both reuse the caller's User-Agent.
	require.Equal(t, "Bearer test-token", gotPollAuth)
	require.Equal(t, "application/json", gotPollAccept)
	require.Empty(t, gotDownloadAuth)
	require.Equal(t, "test-agent/1.0", gotPollUA)
	require.Equal(t, "test-agent/1.0", gotDownloadUA)
}

// TestWaveSpeedPostSubmit_UnknownStatusKeepsPolling exercises the unknown
// terminal-status classification: a status WaveSpeed introduces after this
// SDK shipped must not abort a running task, so an unrecognized status keeps
// polling on the regular schedule until a known terminal status arrives.
func TestWaveSpeedPostSubmit_UnknownStatusKeepsPolling(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var (
		polls     atomic.Int64
		downloads atomic.Int64
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != resultPath {
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})

			return
		}
		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			_, _ = w.Write([]byte(wavespeedTaskBody("queued", resultPath, "", nil)))

			return
		}
		_, _ = w.Write(
			[]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})),
		)
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load(), "an unknown status keeps polling")
	require.Equal(t, int64(1), downloads.Load())
}

// TestWaveSpeedPostSubmit_TransientPollErrors exercises the transient-error
// path: a 429 poll response (without Retry-After) falls back to the regular
// schedule and the exchange completes.
func TestWaveSpeedPostSubmit_TransientPollErrors(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var polls atomic.Int64
	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != resultPath {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})

			return
		}
		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)

			return
		}
		_, _ = w.Write([]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})))
	}))
	defer server.Close()
	outputURL = server.URL + "/output.png"

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load())
}

// TestWaveSpeedPostSubmit_FailedStatus exercises the terminal-failure path: a
// failed task stops polling and surfaces the provider's error message.
//
//nolint:dupl // Mirrors the parallel queue-provider suites; sharing a helper would couple the providers' fixtures.
func TestWaveSpeedPostSubmit_FailedStatus(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		polls.Add(1)
		_, _ = w.Write([]byte(wavespeedTaskBody("failed", resultPath, "CUDA out of memory", nil)))
	}))
	defer server.Close()

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
	require.ErrorContains(t, err, "pred_123")
	require.Equal(t, int64(1), polls.Load(), "a failed task stops polling")
}

// TestWaveSpeedPostSubmit_FailedOnSubmit exercises the submit short-circuit:
// a submit response that already carries a terminal failure status fails
// immediately without issuing any poll request.
func TestWaveSpeedPostSubmit_FailedOnSubmit(t *testing.T) {
	t.Parallel()

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
	}))
	defer server.Close()

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("failed", "", "CUDA out of memory", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
	require.ErrorContains(t, err, "pred_123")
	require.Equal(t, int64(0), polls.Load(), "a failed submit should not poll")
}

// TestWaveSpeedPostSubmit_TerminalFailureStatuses pins the full set of
// WaveSpeed's terminal failure statuses: each one stops the exchange with the
// provider's error without issuing any poll request. ("failed" is covered by
// TestWaveSpeedPostSubmit_FailedOnSubmit.)
func TestWaveSpeedPostSubmit_TerminalFailureStatuses(t *testing.T) {
	t.Parallel()

	for _, status := range []string{"cancelled", "timeout", "deleted"} {
		t.Run(status, func(t *testing.T) {
			t.Parallel()

			var polls atomic.Int64
			server := httptest.NewServer(
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					polls.Add(1)
				}),
			)
			defer server.Close()

			codec := fastImageCodec()
			_, err := codec.PostSubmit(
				context.Background(),
				postSubmitParams(server),
				[]byte(wavespeedTaskBody(status, "", "provider gave up", nil)),
				http.Header{},
			)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
			require.ErrorContains(t, err, "provider gave up")
			require.Equal(t, int64(0), polls.Load(), "a terminal failure never polls")
		})
	}
}

// TestWaveSpeedPostSubmit_ClientErrorPoll exercises the non-transient poll
// failure path: a 4xx poll response stops polling and surfaces the API error.
func TestWaveSpeedPostSubmit_ClientErrorPoll(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	testutils.AssertAPIErrorStatus(t, err, http.StatusUnauthorized)
	require.Equal(t, int64(1), polls.Load(), "a client-error poll stops polling")
}

// TestWaveSpeedPostSubmit_MalformedResponses exercises the malformed-response
// paths: non-JSON task bodies, completed tasks without outputs, missing
// result URLs, and — on the poll side — non-JSON and oversized bodies are
// errors that stop polling.
func TestWaveSpeedPostSubmit_MalformedResponses(t *testing.T) {
	t.Parallel()

	t.Run("non-JSON submit body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
			}),
		)
		defer server.Close()

		codec := fastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte("not json"),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("completed without outputs", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := fastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(wavespeedTaskBody("completed", "", "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without outputs")
	})

	t.Run("missing result URL", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := fastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(wavespeedTaskBody("created", "", "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "no result URL")
	})

	t.Run("missing data object", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"code":200,"message":"success"}`))
			}),
		)
		defer server.Close()

		codec := fastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(`{"code":200,"message":"success"}`),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "missing data object")
	})

	t.Run("non-JSON poll body", func(t *testing.T) {
		t.Parallel()

		const resultPath = "/api/v3/predictions/pred_123/result"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(wavespeedTaskBody("created", resultPath, "", nil)))

				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("not json"))
		}))
		defer server.Close()

		codec := fastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode WaveSpeed task response")
		require.Equal(t, int64(1), polls.Load(), "a malformed poll stops polling")
	})

	t.Run("oversized poll body", func(t *testing.T) {
		t.Parallel()

		const resultPath = "/api/v3/predictions/pred_123/result"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(wavespeedTaskBody("created", resultPath, "", nil)))

				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("processing", resultPath, "", nil)))
		}))
		defer server.Close()

		codec := fastImageCodec()
		params := postSubmitParams(server)
		params.MaxResponseBodyBytes = 4
		_, err := codec.PostSubmit(
			context.Background(),
			params,
			[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "exceeds max size")
		require.Equal(t, int64(1), polls.Load(), "an oversized poll stops polling")
	})
}

// TestWaveSpeedPostSubmit_MediaTypeMismatch exercises the output validation:
// an image codec rejects a non-image output.
func TestWaveSpeedPostSubmit_MediaTypeMismatch(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})))

			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("not an image"))
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	require.ErrorContains(t, err, "image/")
}

// TestWaveSpeedPostSubmit_MediaTypeMismatch_VideoCodec mirrors the image
// media-type-mismatch test for the text-to-video codec: a non-video output is
// rejected.
func TestWaveSpeedPostSubmit_MediaTypeMismatch_VideoCodec(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})))

			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastVideoCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	require.ErrorContains(t, err, "video/")
}

// TestWaveSpeedPostSubmit_MediaTypeMismatch_ImageToImageCodec mirrors the
// image media-type-mismatch test for the image-to-image codec: a non-image
// output is rejected.
func TestWaveSpeedPostSubmit_MediaTypeMismatch_ImageToImageCodec(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})))

			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastImageToImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	require.ErrorContains(t, err, "image/")
}

// TestWaveSpeedPostSubmit_SniffedMediaType exercises the media-type fallback:
// when the CDN omits Content-Type, the bytes are sniffed.
func TestWaveSpeedPostSubmit_SniffedMediaType(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})))

			return
		}
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("completed", "", "", []string{outputURL})),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
}

// TestWaveSpeedPostSubmit_ContextCanceled exercises cancellation: a canceled
// context aborts the exchange with the context error.
func TestWaveSpeedPostSubmit_ContextCanceled(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	codec := fastImageCodec()
	_, err := codec.PostSubmit(
		ctx,
		postSubmitParams(server),
		[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
		http.Header{},
	)
	require.ErrorIs(t, err, context.Canceled)
}

// TestWaveSpeedPostSubmit_ContextCanceledDuringPoll exercises cancellation
// after polling has started: the context is canceled while the second poll
// request is in flight, and the resulting transport error — which wraps
// context.Canceled — must surface immediately instead of being swallowed as
// a transient failure worth retrying.
func TestWaveSpeedPostSubmit_ContextCanceledDuringPoll(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var polls atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// pollStarted is closed when the second poll arrives; release unblocks
	// its handler so the test can cancel the context mid-roundtrip.
	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			// The submit response kicks off polling.
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("created", resultPath, "", nil)))

			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(wavespeedTaskBody("processing", resultPath, "", nil)))
	}))
	defer server.Close()

	type posted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan posted, 1)
	codec := fastImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			postSubmitParams(server),
			[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
			http.Header{},
		)
		results <- posted{resp: resp, err: err}
	}()

	// Cancel the context while the second poll is in flight: the request
	// dies mid-roundtrip, which is exactly the transient-looking transport
	// failure that must not be swallowed.
	<-pollStarted
	cancel()
	close(release)

	result := <-results
	require.ErrorIs(t, result.err, context.Canceled)
	require.Equal(t, int64(2), polls.Load(), "cancellation stops further polling")
}

// TestWaveSpeedPostSubmit_DeadlineExceededDuringPoll exercises deadline expiry
// after polling has started: the caller's deadline lapses while a poll request
// is in flight, and the resulting transport error — which wraps
// context.DeadlineExceeded — must surface immediately instead of being
// swallowed as a transient failure worth retrying. This is the deadline twin
// of TestWaveSpeedPostSubmit_ContextCanceledDuringPoll, pinning that both
// context endings stop the poll loop promptly.
func TestWaveSpeedPostSubmit_DeadlineExceededDuringPoll(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var polls atomic.Int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// pollStarted is closed when the second poll arrives; release unblocks
	// its handler after the deadline has lapsed, so the in-flight poll dies
	// with a context error.
	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			// The submit response kicks off polling.
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(wavespeedTaskBody("created", resultPath, "", nil)))

			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(wavespeedTaskBody("processing", resultPath, "", nil)))
	}))
	defer server.Close()

	type posted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan posted, 1)
	codec := fastImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			postSubmitParams(server),
			[]byte(wavespeedTaskBody("created", resultPath, "", nil)),
			http.Header{},
		)
		results <- posted{resp: resp, err: err}
	}()

	// Let the deadline lapse while the second poll is in flight before
	// releasing the handler: the poll request dies mid-roundtrip, which is
	// exactly the transient-looking transport failure that must not be
	// swallowed.
	<-pollStarted
	time.Sleep(80 * time.Millisecond)
	close(release)

	result := <-results
	require.ErrorIs(t, result.err, context.DeadlineExceeded)
	require.Equal(t, int64(2), polls.Load(), "deadline expiry stops further polling")
}

// TestWaveSpeedPollTarget exercises the poll-target resolution: the Hugging
// Face router gains a /wavespeed prefix, other hosts resolve as-is, an
// absolute URL contributes only its path, queries ride along, and missing or
// unparseable inputs fail with their respective kinds.
func TestWaveSpeedPollTarget(t *testing.T) {
	t.Parallel()

	pollTarget := func(t *testing.T, baseURL, resultPath string) (request.Params, string) {
		t.Helper()

		pollParams, pollPath, err := wavespeedPollTarget(
			PostSubmitParams{BaseURL: baseURL},
			&wavespeedTaskData{URLs: wavespeedTaskURLs{Get: resultPath}},
		)
		require.NoError(t, err)

		return pollParams, pollPath
	}

	t.Run("router host gains /wavespeed prefix", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://router.huggingface.co", "/api/v3/predictions/pred_123/result",
		)
		require.Equal(t, "https://router.huggingface.co/wavespeed", pollParams.BaseURL)
		require.Equal(t, "/api/v3/predictions/pred_123/result", pollPath)
	})

	t.Run("router host matched case-insensitively", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://Router.HuggingFace.CO", "/api/v3/predictions/pred_123/result",
		)
		require.Equal(t, "https://Router.HuggingFace.CO/wavespeed", pollParams.BaseURL)
		require.Equal(t, "/api/v3/predictions/pred_123/result", pollPath)
	})

	t.Run("other hosts resolve as-is", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://api.example.com", "/api/v3/predictions/pred_123/result",
		)
		require.Equal(t, "https://api.example.com", pollParams.BaseURL)
		require.Equal(t, "/api/v3/predictions/pred_123/result", pollPath)
	})

	t.Run("query string appended to the path", func(t *testing.T) {
		t.Parallel()

		_, pollPath := pollTarget(
			t, "https://router.huggingface.co", "/api/v3/predictions/pred_123/result?trace=1",
		)
		require.Equal(t, "/api/v3/predictions/pred_123/result?trace=1", pollPath)
	})

	t.Run("absolute result URL resolved from its path", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t,
			"https://router.huggingface.co",
			"https://api.wavespeed.ai/api/v3/predictions/pred_123/result",
		)
		require.Equal(t, "https://router.huggingface.co/wavespeed", pollParams.BaseURL)
		require.Equal(t, "/api/v3/predictions/pred_123/result", pollPath)
	})

	t.Run("relative result path gains a leading slash", func(t *testing.T) {
		t.Parallel()

		_, pollPath := pollTarget(
			t, "https://router.huggingface.co", "api/v3/predictions/pred_123/result",
		)
		require.Equal(t, "/api/v3/predictions/pred_123/result", pollPath)
	})

	t.Run("empty result path is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := wavespeedPollTarget(
			PostSubmitParams{BaseURL: "https://router.huggingface.co"},
			&wavespeedTaskData{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("invalid base URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := wavespeedPollTarget(
			PostSubmitParams{BaseURL: "http://[::1:bad"},
			&wavespeedTaskData{URLs: wavespeedTaskURLs{Get: "/api/v3/predictions/pred_123/result"}},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("invalid result path is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := wavespeedPollTarget(
			PostSubmitParams{BaseURL: "https://router.huggingface.co"},
			&wavespeedTaskData{URLs: wavespeedTaskURLs{Get: "http://h/%zz"}},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestWaveSpeedPostSubmit_ForeignResultURLCannotRedirect pins the poll
// invariant end to end: a provider-supplied absolute result URL pointing at a
// foreign host cannot redirect the token-bearing poll — the host is discarded
// and the path resolves against the caller's base URL.
func TestWaveSpeedPostSubmit_ForeignResultURLCannotRedirect(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	foreign := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("a provider-supplied absolute result URL must not redirect the poll")
	}))
	defer foreign.Close()

	var (
		polls       atomic.Int64
		downloads   atomic.Int64
		outputURL   string
		gotPollAuth string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case resultPath:
			gotPollAuth = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write([]byte(wavespeedTaskBody("processing", resultPath, "", nil)))

				return
			}
			_, _ = w.Write(
				[]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})),
			)
		case "/cdn/output.png":
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		default:
			t.Errorf("unexpected request path %q", r.URL.Path)
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		// The submit response carries an absolute result URL pointing at a
		// foreign host; the poll must still resolve to the caller's host.
		[]byte(wavespeedTaskBody("created", foreign.URL+resultPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())
	// The token stays on the caller's base host, never the foreign host.
	require.Equal(t, "Bearer test-token", gotPollAuth)
}

// TestWaveSpeedMetadata exercises provider-metadata extraction: an empty task
// yields nil so callers never observe an empty map.
func TestWaveSpeedMetadata(t *testing.T) {
	t.Parallel()

	t.Run("empty task data", func(t *testing.T) {
		t.Parallel()

		require.Nil(t, wavespeedMetadata(&wavespeedTaskData{}))
	})

	t.Run("populated task data", func(t *testing.T) {
		t.Parallel()

		meta := wavespeedMetadata(&wavespeedTaskData{
			ID:        "pred_1",
			Model:     "wavespeed-ai/flux-dev",
			CreatedAt: "2026-01-01T00:00:00Z",
			Timings:   wavespeedTimings{Inference: 2500},
		})
		require.Equal(t, "pred_1", meta["id"])
		require.Equal(t, "wavespeed-ai/flux-dev", meta["model"])
		require.Equal(t, "2026-01-01T00:00:00Z", meta["created_at"])
		require.Equal(t, map[string]any{"inference": float64(2500)}, meta["timings"])
	})

	t.Run("execution time and nsfw flags", func(t *testing.T) {
		t.Parallel()

		meta := wavespeedMetadata(&wavespeedTaskData{
			ID:              "pred_2",
			ExecutionTime:   12.5,
			HasNSFWContents: []bool{false},
		})
		require.InEpsilon(t, 12.5, meta["execution_time"], 0.001)
		require.Equal(t, []bool{false}, meta["has_nsfw_contents"])
	})
}

// TestWaveSpeedTaskFailedError exercises the terminal-failure error message:
// the provider's error message and task ID are appended only when present.
func TestWaveSpeedTaskFailedError(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		data wavespeedTaskData
		want string
	}{
		{
			name: "bare failure",
			data: wavespeedTaskData{Status: "failed"},
			want: "WaveSpeed task failed",
		},
		{
			name: "with provider error",
			data: wavespeedTaskData{Status: "failed", Error: "CUDA out of memory"},
			want: "WaveSpeed task failed: CUDA out of memory",
		},
		{
			name: "with task id",
			data: wavespeedTaskData{Status: "failed", ID: "pred_1"},
			want: "WaveSpeed task failed (task id pred_1)",
		},
		{
			name: "with provider error and task id",
			data: wavespeedTaskData{
				Status: "failed",
				ID:     "pred_1",
				Error:  "CUDA out of memory",
			},
			want: "WaveSpeed task failed: CUDA out of memory (task id pred_1)",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := wavespeedTaskFailedError(&tc.data)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

// TestWaveSpeedCompletedWithOutputs exercises the completed-task output
// validation: valid outputs return the task data, while an empty output list
// or an empty first output is a serialization error.
func TestWaveSpeedCompletedWithOutputs(t *testing.T) {
	t.Parallel()

	t.Run("with outputs", func(t *testing.T) {
		t.Parallel()

		data := wavespeedTaskData{
			Status:  "completed",
			Outputs: []string{"https://cdn.example.com/output.png"},
		}
		out, err := wavespeedCompletedWithOutputs(&data)
		require.NoError(t, err)
		require.Same(t, &data, out)
	})

	t.Run("without outputs", func(t *testing.T) {
		t.Parallel()

		_, err := wavespeedCompletedWithOutputs(&wavespeedTaskData{Status: "completed"})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without outputs")
	})

	t.Run("empty first output", func(t *testing.T) {
		t.Parallel()

		_, err := wavespeedCompletedWithOutputs(&wavespeedTaskData{
			Status:  "completed",
			Outputs: []string{""},
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without outputs")
	})
}

// TestWaveSpeedCodec_EncodeUnmarshalableProviderParams exercises every task
// codec's encode-side marshal guard: a ProviderParams entry json cannot
// encode makes the whole flat body unmarshalable, which surfaces as a
// serialization error instead of a garbled request.
func TestWaveSpeedCodec_EncodeUnmarshalableProviderParams(t *testing.T) {
	t.Parallel()

	t.Run("text-to-video", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedTextToVideoCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "wavespeed-ai/minimax-h3/text-to-video",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("text-to-image", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedTextToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "wavespeed-ai/flux-dev",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("image-to-image", func(t *testing.T) {
		t.Parallel()

		codec := wavespeedImageToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{
				Input:          "data:image/png;base64,abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "bria/fibo-2.0/edit",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})
}

// TestWaveSpeedPostSubmit_DownloadBodyLimit exercises the download-side
// response-size cap: a completed task whose CDN output exceeds
// PostSubmitParams.MaxResponseBodyBytes fails instead of downloading without
// bound.
func TestWaveSpeedPostSubmit_DownloadBodyLimit(t *testing.T) {
	t.Parallel()

	const resultPath = "/api/v3/predictions/pred_123/result"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(
				[]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})),
			)

			return
		}

		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', 0, 0, 0, 0, 0, 0, 0, 0})
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastImageCodec()
	params := postSubmitParams(server)
	params.MaxResponseBodyBytes = 4
	_, err := codec.PostSubmit(
		context.Background(),
		params,
		[]byte(wavespeedTaskBody("completed", resultPath, "", []string{outputURL})),
		http.Header{},
	)
	require.Error(t, err)
}
