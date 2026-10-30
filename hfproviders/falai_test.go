//go:build !integration

package hfproviders

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// falaiTestTaskPath is the path component of the test task's response URL.
// fal hosts response URLs on its own queue host; the exchange uses only the
// path, so follow-up requests target the caller's base URL (the test server).
const falaiTestTaskPath = "/fal-ai/flux/requests/req_123"

// falaiState builds a fal queue state body for the given status, carrying the
// standard test request ID and a response URL on fal's queue host. fields
// merge over the base state so tests can add error, metrics, or position
// entries.
func falaiState(status string, fields map[string]any) string {
	state := map[string]any{
		"request_id":   "req_123",
		"status":       status,
		"response_url": "https://queue.fal.run" + falaiTestTaskPath,
	}
	maps.Copy(state, fields)

	body, err := json.Marshal(state)
	if err != nil {
		// Unreachable: every value above marshals without error.
		panic(err)
	}

	return string(body)
}

// falaiStateBody builds a fal queue state body, optionally carrying the
// provider's failure message and category.
func falaiStateBody(status, taskErr, errType string) string {
	fields := map[string]any{}
	if taskErr != "" {
		fields["error"] = taskErr
	}
	if errType != "" {
		fields["error_type"] = errType
	}

	return falaiState(status, fields)
}

// falaiPostSubmitParams builds transport parameters targeting the given test
// server.
func falaiPostSubmitParams(server *httptest.Server) PostSubmitParams {
	return PostSubmitParams{
		HTTPClient:           server.Client(),
		BaseURL:              server.URL,
		Token:                "test-token",
		UserAgent:            "test-agent/1.0",
		MaxResponseBodyBytes: 1 << 20,
	}
}

// falaiFastImageCodec returns a text-to-image codec with fast polling, so
// tests exercise the exchange's branching without real delays.
func falaiFastImageCodec() falaiTextToImageCodec {
	return falaiTextToImageCodec{falaiQueueCodec{
		mediaPrefix:     falaiImageMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}
}

// falaiFastVideoCodec mirrors falaiFastImageCodec for the text-to-video
// prefix.
func falaiFastVideoCodec() falaiTextToVideoCodec {
	return falaiTextToVideoCodec{falaiQueueCodec{
		mediaPrefix:     falaiVideoMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}
}

// falaiFastImageToImageCodec mirrors falaiFastImageCodec for image-to-image.
func falaiFastImageToImageCodec() falaiImageToImageCodec {
	return falaiImageToImageCodec{falaiQueueCodec{
		mediaPrefix:     falaiImageMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}
}

// falaiFastSegmentationCodec mirrors falaiFastImageCodec for image
// segmentation.
func falaiFastSegmentationCodec() falaiImageSegmentationCodec {
	return falaiImageSegmentationCodec{falaiQueueCodec{
		mediaPrefix:     falaiImageMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}
}

// falaiFastSpeechCodec returns a speech-recognition codec with fast polling.
func falaiFastSpeechCodec() falaiSpeechRecognitionCodec {
	return falaiSpeechRecognitionCodec{falaiQueueCodec{
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}
}

func TestFalAIProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewFalAIProvider(mapper)
	require.NoError(t, err)

	require.Equal(t, "fal-ai", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

func TestNewFalAIProvider_NilResolver(t *testing.T) {
	t.Parallel()

	p, err := NewFalAIProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, FalAIProvider{}, p)
}

func TestFalAIProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var p FalAIProvider

	endpoints := []func(EndpointParams) (Endpoint, error){
		p.TextToImageEndpoint,
		p.ImageToImageEndpoint,
		p.TextToVideoEndpoint,
		p.ImageSegmentationEndpoint,
		p.SpeechRecognitionEndpoint,
	}
	for _, endpoint := range endpoints {
		_, err := endpoint(EndpointParams{
			Context: context.Background(),
			Model:   "any-model",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	}
}

// falaiEndpointCase pairs a display name with the endpoint method it
// exercises and the fal mapping that method must resolve to.
type falaiEndpointCase struct {
	name            string
	task            Task
	model           string
	providerModelID string
	endpoint        func(FalAIProvider, EndpointParams) (Endpoint, error)
}

func TestFalAIProvider_Endpoints(t *testing.T) {
	t.Parallel()

	cases := []falaiEndpointCase{
		{
			name:            "TextToImageEndpoint",
			task:            TaskTextToImage,
			model:           "krea/Krea-2-Turbo",
			providerModelID: "fal-ai/krea-2/turbo",
			endpoint:        FalAIProvider.TextToImageEndpoint,
		},
		{
			name:            "ImageToImageEndpoint",
			task:            TaskImageToImage,
			model:           "black-forest-labs/FLUX.2-klein-9B",
			providerModelID: "fal-ai/flux-2/klein/9b/edit",
			endpoint:        FalAIProvider.ImageToImageEndpoint,
		},
		{
			name:            "TextToVideoEndpoint",
			task:            TaskTextToVideo,
			model:           "Wan-AI/Wan2.2-TI2V-5B",
			providerModelID: "fal-ai/wan/v2.2-ti2v-5b/text-to-video",
			endpoint:        FalAIProvider.TextToVideoEndpoint,
		},
		{
			name:            "ImageSegmentationEndpoint",
			task:            TaskImageSegmentation,
			model:           "briaai/RMBG-2.0",
			providerModelID: "fal-ai/bria/background/remove",
			endpoint:        FalAIProvider.ImageSegmentationEndpoint,
		},
		{
			name:            "SpeechRecognitionEndpoint",
			task:            TaskAutomaticSpeechRecognition,
			model:           "openai/whisper-large-v3",
			providerModelID: "fal-ai/whisper",
			endpoint:        FalAIProvider.SpeechRecognitionEndpoint,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: "fal-ai",
					Task:     tc.task,
					Model:    tc.model,
				}: {ProviderModelID: tc.providerModelID},
			})
			p, err := NewFalAIProvider(mapper)
			require.NoError(t, err)

			endpoint, err := tc.endpoint(p, EndpointParams{
				Context: context.Background(),
				Model:   tc.model,
			})
			require.NoError(t, err)
			require.Equal(t, http.MethodPost, endpoint.Method)
			// The router prefix and fal's own "fal-ai/" namespace deliberately
			// repeat the segment, matching the official Hugging Face clients.
			require.Equal(t,
				"fal-ai/"+tc.providerModelID+"?_subdomain=queue",
				endpoint.Path,
			)
			require.NotNil(t, endpoint.Mapping)
			require.Equal(t, tc.providerModelID, endpoint.Mapping.ProviderModelID)
		})
	}
}

func TestFalAIProvider_Endpoint_ResolutionError(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewFalAIProvider(mapper)
	require.NoError(t, err)

	_, err = p.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "unknown-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// falaiRequireFlatEncode asserts an encoder produced the expected flat fal.ai
// body with JSON content headers.
func falaiRequireFlatEncode[Req any](
	t *testing.T,
	encode func(EncodeParams[Req]) ([]byte, http.Header, error),
	request Req,
	wantJSON string,
) {
	t.Helper()

	body, headers, err := encode(EncodeParams[Req]{
		Context: context.Background(),
		Request: request,
	})
	require.NoError(t, err)
	require.JSONEq(t, wantJSON, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "application/json", headers.Get("Accept"))
}

func TestFalAIProvider_TextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := falaiTextToImageCodec{}

	t.Run("flat body with parameters", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					GuidanceScale:     new(7.5),
					NumInferenceSteps: new(30),
					Seed:              new(int64(42)),
				},
			},
			`{"prompt":"a cat","guidance_scale":7.5,"num_inference_steps":30,"seed":42}`,
		)
	})

	t.Run("nil parameters yields prompt only", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			`{"prompt":"a cat"}`,
		)
	})

	t.Run("width and height fold into image_size", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					Width:  new(512),
					Height: new(768),
				},
			},
			`{"prompt":"a cat","image_size":{"width":512,"height":768}}`,
		)
	})

	t.Run("a lone width stays flat", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input:      "a cat",
				Parameters: &hftypes.TextToImageParameters{Width: new(512)},
			},
			`{"prompt":"a cat","width":512}`,
		)
	})

	t.Run("provider params merged and override typed fields", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					Width:  new(512),
					Height: new(768),
				},
				ProviderParams: map[string]any{
					"enable_safety_checker": false,
					"image_size":            "square_hd",
				},
			},
			`{
				"prompt": "a cat",
				"image_size": "square_hd",
				"enable_safety_checker": false
			}`,
		)
	})
}

func TestFalAIProvider_TextToVideoCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := falaiTextToVideoCodec{}

	t.Run("flat body with parameters", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
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

	t.Run("provider params merged and override typed fields", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToVideoParameters{
					NumFrames: new(24),
				},
				ProviderParams: map[string]any{
					"duration":   5,
					"num_frames": 48,
				},
			},
			`{"prompt":"a cat","num_frames":48,"duration":5}`,
		)
	})
}

func TestFalAIProvider_ImageToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := falaiImageToImageCodec{}

	t.Run("data URI passes through on both fields", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				Parameters: &hftypes.ImageToImageParameters{
					Prompt: new("make it blue"),
				},
			},
			`{
				"image_url": "data:image/png;base64,abc123",
				"image_urls": ["data:image/png;base64,abc123"],
				"prompt": "make it blue"
			}`,
		)
	})

	t.Run("http URL passes through", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "https://example.com/cat.png",
			},
			`{
				"image_url": "https://example.com/cat.png",
				"image_urls": ["https://example.com/cat.png"]
			}`,
		)
	})

	t.Run("raw base64 wrapped as jpeg data URI", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{Input: "abc123"},
			`{
				"image_url": "data:image/jpeg;base64,abc123",
				"image_urls": ["data:image/jpeg;base64,abc123"]
			}`,
		)
	})

	t.Run("target_size renames to image_size", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "abc123",
				Parameters: &hftypes.ImageToImageParameters{
					TargetSize: &hftypes.ImageToImageTargetSize{
						Width:  new(256),
						Height: new(256),
					},
				},
			},
			`{
				"image_url": "data:image/jpeg;base64,abc123",
				"image_urls": ["data:image/jpeg;base64,abc123"],
				"image_size": {"width": 256, "height": 256}
			}`,
		)
	})

	t.Run("provider params override the image fields", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				ProviderParams: map[string]any{
					"image_urls": []string{
						"https://example.com/one.png",
						"https://example.com/two.png",
					},
				},
			},
			`{
				"image_url": "data:image/png;base64,abc123",
				"image_urls": ["https://example.com/one.png", "https://example.com/two.png"]
			}`,
		)
	})
}

func TestFalAIProvider_ImageSegmentationCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := falaiImageSegmentationCodec{}

	t.Run("raw base64 wrapped as png data URI", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageSegmentationRequest{Input: "abc123"},
			`{
				"image_url": "data:image/png;base64,abc123",
				"sync_mode": true
			}`,
		)
	})

	t.Run("parameters merged", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageSegmentationRequest{
				Input: "https://example.com/cat.png",
				Parameters: &hftypes.ImageSegmentationParameters{
					Threshold: new(0.5),
					Subtask:   hftypes.ImageSegmentationSubtaskSemantic,
				},
			},
			`{
				"image_url": "https://example.com/cat.png",
				"threshold": 0.5,
				"subtask": "semantic",
				"sync_mode": true
			}`,
		)
	})

	t.Run("sync_mode is forced even over provider params", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.ImageSegmentationRequest{
				Input:          "abc123",
				ProviderParams: map[string]any{"sync_mode": false},
			},
			`{
				"image_url": "data:image/png;base64,abc123",
				"sync_mode": true
			}`,
		)
	})
}

func TestFalAIProvider_SpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := falaiSpeechRecognitionCodec{}

	t.Run("raw base64 wrapped as mpeg data URI", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{Input: "abc123"},
			`{"audio_url":"data:audio/mpeg;base64,abc123"}`,
		)
	})

	t.Run("http URL passes through", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "https://example.com/sample.flac",
			},
			`{"audio_url":"https://example.com/sample.flac"}`,
		)
	})

	t.Run("data URI passes through", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "data:audio/x-wav;base64,abc123",
			},
			`{"audio_url":"data:audio/x-wav;base64,abc123"}`,
		)
	})

	t.Run("parameters and provider params merged", func(t *testing.T) {
		t.Parallel()

		falaiRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "abc123",
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
				ProviderParams: map[string]any{
					"num_beams": 5,
				},
			},
			`{
				"audio_url": "data:audio/mpeg;base64,abc123",
				"return_timestamps": true,
				"num_beams": 5
			}`,
		)
	})
}

// TestFalAIProvider_AdapterLorasInjection exercises the LoRA payload
// injection: a LoRA-adapter mapping injects the "loras" entry naming the
// adapter weights' Hub resolve URL, the generic SD-class LoRA endpoint gains its
// base model name on text-to-image only, a caller-supplied "loras" wins,
// non-adapter models and nil resolvers leave the payload untouched, and
// lookup failures propagate.
func TestFalAIProvider_AdapterLorasInjection(t *testing.T) {
	t.Parallel()

	t.Run("injected on the lora-capable tasks", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToVideo,
				Model:    "ZhengmingYu/DMAD",
			}: {
				ProviderModelID:    "fal-ai/ltx-2-19b/text-to-video/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "dmad/dmad.safetensors",
				Status:             "live",
			},
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "some-lora/flux-lora",
			}: {
				ProviderModelID:    "fal-ai/flux-dev",
				Adapter:            "lora",
				AdapterWeightsPath: "weights/lora.safetensors",
			},
			{
				Provider: "fal-ai",
				Task:     TaskImageToImage,
				Model:    "some-lora/flux-lora",
			}: {
				ProviderModelID:    "fal-ai/flux-dev",
				Adapter:            "lora",
				AdapterWeightsPath: "weights/lora.safetensors",
			},
		})

		video := FalAITextToVideoMixin{Resolver: resolver}.TextToVideoCodec()
		body, _, err := video.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/ZhengmingYu/DMAD/resolve/main/dmad/dmad.safetensors",
				"scale":1
			}]
		}`, string(body))

		image := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()
		body, _, err = image.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/some-lora/flux-lora/resolve/main/weights/lora.safetensors",
				"scale":1
			}]
		}`, string(body))

		i2i := FalAIImageToImageMixin{Resolver: resolver}.ImageToImageCodec()
		body, _, err = i2i.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{Input: "abc123"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"image_url": "data:image/jpeg;base64,abc123",
			"image_urls": ["data:image/jpeg;base64,abc123"],
			"loras": [{
				"path": "https://huggingface.co/some-lora/flux-lora/resolve/main/weights/lora.safetensors",
				"scale": 1
			}]
		}`, string(body))
	})

	t.Run("generic lora endpoint gains the base model on text-to-image", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "user/sdxl-lora",
			}: {
				ProviderModelID:    "fal-ai/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "lora.safetensors",
			},
			{
				Provider: "fal-ai",
				Task:     TaskTextToVideo,
				Model:    "user/sdxl-lora",
			}: {
				ProviderModelID:    "fal-ai/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "lora.safetensors",
			},
		})

		image := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()
		body, _, err := image.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "user/sdxl-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/user/sdxl-lora/resolve/main/lora.safetensors",
				"scale":1
			}],
			"model_name":"stabilityai/stable-diffusion-xl-base-1.0"
		}`, string(body))

		// The same mapping on another task must not gain the base model:
		// fal only requires it for the SD-class LoRA endpoint.
		video := FalAITextToVideoMixin{Resolver: resolver}.TextToVideoCodec()
		body, _, err = video.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "user/sdxl-lora",
		})
		require.NoError(t, err)
		require.NotContains(t, string(body), "model_name")
	})

	t.Run("caller-supplied model_name wins on the generic lora endpoint", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "user/sd15-lora",
			}: {
				ProviderModelID:    "fal-ai/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "lora.safetensors",
			},
		})
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					"model_name": "stabilityai/stable-diffusion-1-5",
				},
			},
			Model: "user/sd15-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/user/sd15-lora/resolve/main/lora.safetensors",
				"scale":1
			}],
			"model_name":"stabilityai/stable-diffusion-1-5"
		}`, string(body))
	})

	t.Run("caller-supplied loras win", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToVideo,
				Model:    "ZhengmingYu/DMAD",
			}: {
				ProviderModelID:    "fal-ai/ltx-2-19b/text-to-video/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "dmad/dmad.safetensors",
			},
		})
		codec := FalAITextToVideoMixin{Resolver: resolver}.TextToVideoCodec()

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

	t.Run("caller-supplied loras keep the base-model default", func(t *testing.T) {
		t.Parallel()

		// The two injected entries are independent: overriding "loras" on the
		// generic SD-class LoRA endpoint still keeps the "model_name" default
		// fal requires for that endpoint, so the request does not 4xx.
		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "user/sdxl-lora",
			}: {
				ProviderModelID:    "fal-ai/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "lora.safetensors",
			},
		})
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					"loras": []map[string]any{{"path": "user/own-lora", "scale": 0.8}},
				},
			},
			Model: "user/sdxl-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{"path":"user/own-lora","scale":0.8}],
			"model_name":"stabilityai/stable-diffusion-xl-base-1.0"
		}`, string(body))
	})

	t.Run("non-adapter mapping injects nothing", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "krea/Krea-2-Turbo",
			}: {ProviderModelID: "fal-ai/krea-2/turbo"},
		})
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "krea/Krea-2-Turbo",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"prompt":"a cat"}`, string(body))
	})

	t.Run("adapter mapping without weights path injects nothing", func(t *testing.T) {
		t.Parallel()

		resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
			{
				Provider: "fal-ai",
				Task:     TaskTextToImage,
				Model:    "some-lora/flux-lora",
			}: {ProviderModelID: "fal-ai/flux-dev", Adapter: "lora"},
		})
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"prompt":"a cat"}`, string(body))
	})

	t.Run("nil resolver injects nothing", func(t *testing.T) {
		t.Parallel()

		codec := falaiTextToImageCodec{}

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"prompt":"a cat"}`, string(body))
	})

	t.Run("lookup failure propagates", func(t *testing.T) {
		t.Parallel()

		codec := FalAITextToImageMixin{Resolver: NewStaticModelResolver(nil)}.TextToImageCodec()

		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "unknown-model",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

// TestFalAIProvider_EncodePrefersEndpointMapping verifies that Encode reuses
// the mapping resolved at endpoint-construction time
// ([EncodeParams.Mapping]) instead of looking it up again: the LoRA entry is
// injected from the supplied mapping while the codec's resolver is never
// called. When no mapping is supplied, the resolver is consulted exactly
// once.
func TestFalAIProvider_EncodePrefersEndpointMapping(t *testing.T) {
	t.Parallel()

	t.Run("supplied mapping used without resolver call", func(t *testing.T) {
		t.Parallel()

		// The resolver would answer with a non-adapter mapping: if Encode
		// consulted it, no loras entry would appear in the body.
		resolver := &countingResolver{
			mapping: ModelMapping{ProviderModelID: "fal-ai/flux-dev"},
		}
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "ZhengmingYu/DMAD",
			Mapping: &ModelMapping{
				ProviderModelID:    "fal-ai/ltx-2-19b/text-to-video/lora",
				Adapter:            "lora",
				AdapterWeightsPath: "dmad/dmad.safetensors",
			},
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/ZhengmingYu/DMAD/resolve/main/dmad/dmad.safetensors",
				"scale":1
			}]
		}`, string(body))
		require.Zero(t, resolver.calls.Load())
	})

	t.Run("nil mapping falls back to resolver", func(t *testing.T) {
		t.Parallel()

		resolver := &countingResolver{
			mapping: ModelMapping{
				ProviderModelID:    "fal-ai/flux-dev",
				Adapter:            "lora",
				AdapterWeightsPath: "weights/lora.safetensors",
			},
		}
		codec := FalAITextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
		})
		require.NoError(t, err)
		require.JSONEq(t, `{
			"prompt":"a cat",
			"loras":[{
				"path":"https://huggingface.co/some-lora/flux-lora/resolve/main/weights/lora.safetensors",
				"scale":1
			}]
		}`, string(body))
		require.Equal(t, int64(1), resolver.calls.Load())
	})
}

// TestFalAIProvider_CodecDecodes exercises the media-output codecs shared by
// the text-to-image, image-to-image, and text-to-video mixins.
func TestFalAIProvider_CodecDecodes(t *testing.T) {
	t.Parallel()

	t.Run("text-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := falaiTextToImageCodec{falaiQueueCodec{mediaPrefix: falaiImageMediaPrefix}}
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.TextToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})

	t.Run("image-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := falaiImageToImageCodec{falaiQueueCodec{mediaPrefix: falaiImageMediaPrefix}}
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.ImageToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})

	t.Run("text-to-video codec", func(t *testing.T) {
		t.Parallel()

		codec := falaiTextToVideoCodec{falaiQueueCodec{mediaPrefix: falaiVideoMediaPrefix}}

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
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		})
	})
}

func TestFalAIProvider_ImageSegmentationCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := falaiImageSegmentationCodec{falaiQueueCodec{mediaPrefix: falaiImageMediaPrefix}}

	t.Run("valid mask image", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "image/png")
		body := []byte{0x89, 'P', 'N', 'G'}

		segments, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    body,
			Headers: headers,
		})
		require.NoError(t, err)
		require.Len(t, segments, 1)
		require.Equal(t, "mask", segments[0].Label)
		require.Equal(t, "iVBORw==", segments[0].Mask)
		require.Nil(t, segments[0].Score)
	})

	t.Run("invalid content type", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "video/mp4")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			Headers: headers,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

func TestFalAIProvider_SpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := falaiSpeechRecognitionCodec{}

	t.Run("whisper text with chunks", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body: []byte(
				`{"text":"hello world","chunks":[{"text":"hello","timestamp":[0.0,1.0]},` +
					`{"text":"world","timestamp":[1.0,2.0]}]}`,
			),
			Headers: http.Header{"Content-Type": {"application/json"}},
		})
		require.NoError(t, err)
		require.Equal(t, "hello world", resp.Text)
		require.Equal(t, []hftypes.SpeechRecognitionChunk{
			{Text: "hello", Timestamp: []float64{0.0, 1.0}},
			{Text: "world", Timestamp: []float64{1.0, 2.0}},
		}, resp.Chunks)
	})

	t.Run("nemotron output with segments", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body: []byte(
				`{"output":"hello","segments":[{"text":"hello","start":0.5,"end":1.5},` +
					`{"text":"open"}]}`,
			),
			Headers: http.Header{"Content-Type": {"application/json"}},
		})
		require.NoError(t, err)
		require.Equal(t, "hello", resp.Text)
		require.Equal(t, []hftypes.SpeechRecognitionChunk{
			{Text: "hello", Timestamp: []float64{0.5, 1.5}},
			{Text: "open", Timestamp: []float64{0, 0}},
		}, resp.Chunks)
	})

	t.Run("empty transcript string is kept", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":""}`),
			Headers: http.Header{"Content-Type": {"application/json"}},
		})
		require.NoError(t, err)
		require.Empty(t, resp.Text)
		require.Nil(t, resp.Chunks)
	})

	t.Run("missing transcript", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"chunks":[]}`),
			Headers: http.Header{"Content-Type": {"application/json"}},
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "no transcript")
	})

	t.Run("non-JSON body", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte("not json"),
			Headers: http.Header{"Content-Type": {"application/json"}},
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// falaiSubsetProvider serves only text-to-image by embedding the
// corresponding fal.ai mixin with a resolver set directly, without going
// through NewFalAIProvider.
type falaiSubsetProvider struct {
	FalAITextToImageMixin
}

func (falaiSubsetProvider) ProviderSuffix() string { return "" }

func (falaiSubsetProvider) Name() string { return "subset" }

func TestFalAIMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	p := falaiSubsetProvider{
		FalAITextToImageMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: "fal-ai",
					Task:     TaskTextToImage,
					Model:    "krea/Krea-2-Turbo",
				}: {ProviderModelID: "fal-ai/krea-2/turbo"},
			}),
		},
	}

	prov, err := AsProvider[TextToImageProvider](p)
	require.NoError(t, err)

	endpoint, err := prov.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "krea/Krea-2-Turbo",
	})
	require.NoError(t, err)
	require.Equal(t, "fal-ai/fal-ai/krea-2/turbo?_subdomain=queue", endpoint.Path)

	_, err = AsProvider[TextToVideoProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.TextToVideoProvider",
	)

	_, err = AsProvider[SpeechRecognitionProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.SpeechRecognitionProvider",
	)
}

func TestFalAIMixins_ZeroValueResolvers(t *testing.T) {
	t.Parallel()

	var (
		imageMixin  FalAITextToImageMixin
		i2iMixin    FalAIImageToImageMixin
		videoMixin  FalAITextToVideoMixin
		segMixin    FalAIImageSegmentationMixin
		speechMixin FalAISpeechRecognitionMixin
	)

	_, err := imageMixin.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = i2iMixin.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = videoMixin.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = segMixin.ImageSegmentationEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = speechMixin.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestFalAIMixins_CodecsAsync asserts every fal.ai mixin serves an
// AsyncCodec, so the task pipeline routes their responses through PostSubmit.
func TestFalAIMixins_CodecsAsync(t *testing.T) {
	t.Parallel()

	p, err := NewFalAIProvider(NewStaticModelResolver(nil))
	require.NoError(t, err)

	imageAsync, ok := any(p.TextToImageCodec()).(AsyncCodec[hftypes.TextToImageRequest, hftypes.TextToImageResponse])
	require.True(t, ok, "text-to-image codec must implement AsyncCodec")

	i2iAsync, ok := any(p.ImageToImageCodec()).(AsyncCodec[hftypes.ImageToImageRequest, hftypes.ImageToImageResponse])
	require.True(t, ok, "image-to-image codec must implement AsyncCodec")

	videoAsync, ok := any(p.TextToVideoCodec()).(AsyncCodec[hftypes.TextToVideoRequest, hftypes.TextToVideoResponse])
	require.True(t, ok, "text-to-video codec must implement AsyncCodec")

	segAsync, ok := any(p.ImageSegmentationCodec()).(AsyncCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation])
	require.True(t, ok, "image-segmentation codec must implement AsyncCodec")

	speechAsync, ok := any(p.SpeechRecognitionCodec()).(AsyncCodec[hftypes.SpeechRecognitionRequest, hftypes.SpeechRecognitionResponse])
	require.True(t, ok, "speech-recognition codec must implement AsyncCodec")

	require.NotNil(t, imageAsync)
	require.NotNil(t, i2iAsync)
	require.NotNil(t, videoAsync)
	require.NotNil(t, segAsync)
	require.NotNil(t, speechAsync)
}

// TestFalAIPostSubmit_CompleteOnSubmit exercises the submit short-circuit: a
// submit response that already carries the COMPLETED status fetches the
// result and downloads the output without any polling requests.
func TestFalAIPostSubmit_CompleteOnSubmit(t *testing.T) {
	t.Parallel()

	var (
		polls         atomic.Int64
		downloads     atomic.Int64
		gotDownloadUA string
	)
	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		case "/cdn/output.png":
			downloads.Add(1)
			gotDownloadUA = r.Header.Get("User-Agent")
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		default:
			polls.Add(1)
			t.Errorf("unexpected request path %q", r.URL.Path)
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("COMPLETED", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(0), polls.Load(), "a completed submit response needs no polling")
	require.Equal(t, int64(1), downloads.Load())
	require.Equal(t, "test-agent/1.0", gotDownloadUA)
	require.Equal(t, "req_123", resp.ProviderMetadata["request_id"])
}

// TestFalAIPostSubmit_PollUntilComplete exercises the full exchange when the
// task is still running on submit: the status URL is polled (with the
// caller's bearer token) until the task completes, the result is fetched, and
// the output is downloaded unauthenticated from the provider CDN.
func TestFalAIPostSubmit_PollUntilComplete(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var (
		polls           atomic.Int64
		results         atomic.Int64
		downloads       atomic.Int64
		outputURL       string
		gotPollAuth     string
		gotPollAccept   string
		gotPollUA       string
		gotResultAuth   string
		gotDownloadAuth string
		gotDownloadUA   string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			gotPollAuth = r.Header.Get("Authorization")
			gotPollAccept = r.Header.Get("Accept")
			gotPollUA = r.Header.Get("User-Agent")

			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write([]byte(falaiStateBody("IN_QUEUE", "", "")))

				return
			}
			_, _ = w.Write([]byte(falaiState("COMPLETED", map[string]any{
				"metrics": map[string]any{"inference_time": 1.5},
			})))
		case falaiTestTaskPath:
			gotResultAuth = r.Header.Get("Authorization")
			results.Add(1)

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
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

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, "req_123", resp.ProviderMetadata["request_id"])
	require.InEpsilon(t, 1.5, resp.ProviderMetadata["inference_time"], 0.001)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), results.Load())
	require.Equal(t, int64(1), downloads.Load())

	// Polls and the result fetch carry the caller's bearer token and JSON
	// accept; CDN downloads are unauthenticated. All reuse the caller's
	// User-Agent.
	require.Equal(t, "Bearer test-token", gotPollAuth)
	require.Equal(t, "application/json", gotPollAccept)
	require.Equal(t, "Bearer test-token", gotResultAuth)
	require.Empty(t, gotDownloadAuth)
	require.Equal(t, "test-agent/1.0", gotPollUA)
	require.Equal(t, "test-agent/1.0", gotDownloadUA)
}

// TestFalAIPostSubmit_FailedTask exercises the terminal-failure path: a task
// that completes with an error stops the exchange and surfaces the provider's
// error without fetching the result.
func TestFalAIPostSubmit_FailedTask(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var (
		polls   atomic.Int64
		results atomic.Int64
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				falaiStateBody("COMPLETED", "CUDA out of memory", "runner_error"),
			))
		default:
			results.Add(1)
		}
	}))
	defer server.Close()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
	require.ErrorContains(t, err, "runner_error")
	require.ErrorContains(t, err, "req_123")
	require.Equal(t, int64(1), polls.Load(), "a failed task stops polling")
	require.Equal(t, int64(0), results.Load(), "a failed task never fetches the result")
}

// TestFalAIPostSubmit_FailedOnSubmit exercises the submit short-circuit: a
// submit response that already carries the failure error fails immediately
// without issuing any poll request.
func TestFalAIPostSubmit_FailedOnSubmit(t *testing.T) {
	t.Parallel()

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
	}))
	defer server.Close()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("COMPLETED", "provider gave up", "")),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "provider gave up")
	require.Equal(t, int64(0), polls.Load(), "a failed submit should not poll")
}

// TestFalAIPostSubmit_UnknownStatusKeepsPolling exercises the unknown-status
// classification: a status fal introduces after this SDK shipped must not
// abort a running task, so an unrecognized status keeps polling on the
// regular schedule until a terminal status arrives.
func TestFalAIPostSubmit_UnknownStatusKeepsPolling(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var (
		polls     atomic.Int64
		downloads atomic.Int64
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write([]byte(falaiStateBody("PAUSED_BY_PROVIDER", "", "")))

				return
			}
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load(), "an unknown status keeps polling")
	require.Equal(t, int64(1), downloads.Load())
}

// TestFalAIPostSubmit_TransientPollErrors exercises the transient-error
// path: a 429 poll response (without Retry-After) falls back to the regular
// schedule and the exchange completes.
func TestFalAIPostSubmit_TransientPollErrors(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var polls atomic.Int64
	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				w.WriteHeader(http.StatusTooManyRequests)

				return
			}
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load())
}

// TestFalAIPostSubmit_RetryAfterHonored exercises the server-provided wait
// override: a 429 poll carrying Retry-After replaces the regular schedule for
// the following round.
func TestFalAIPostSubmit_RetryAfterHonored(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var (
		polls     atomic.Int64
		downloads atomic.Int64
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			switch polls.Add(1) {
			case 1:
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)

				return
			case 2:
				_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))

				return
			}
			t.Error("polling continued past the completed status")
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())
}

// TestFalAIPostSubmit_ClientErrorPoll exercises the non-transient poll
// failure path: a 4xx poll response stops polling and surfaces the API error.
func TestFalAIPostSubmit_ClientErrorPoll(t *testing.T) {
	t.Parallel()

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	testutils.AssertAPIErrorStatus(t, err, http.StatusUnauthorized)
	require.Equal(t, int64(1), polls.Load(), "a client-error poll stops polling")
}

// TestFalAIPostSubmit_ResultError exercises the failed-task fallback: a task
// whose status carried no error surfaces as the result endpoint's error
// response, which propagates as an APIError.
func TestFalAIPostSubmit_ResultError(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		default:
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(
				`{"detail":"prompt: Field required","error_type":"validation_error"}`,
			))
		}
	}))
	defer server.Close()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	testutils.AssertAPIErrorStatus(t, err, http.StatusUnprocessableEntity)
}

// TestFalAIPostSubmit_MalformedResponses exercises the malformed-response
// paths: non-JSON queue bodies, missing request IDs and response URLs, and —
// on the poll side — non-JSON and oversized bodies are errors that stop
// polling.
func TestFalAIPostSubmit_MalformedResponses(t *testing.T) {
	t.Parallel()

	t.Run("non-JSON submit body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
			}),
		)
		defer server.Close()

		codec := falaiFastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			falaiPostSubmitParams(server),
			[]byte("not json"),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("missing request ID", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := falaiFastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			falaiPostSubmitParams(server),
			[]byte(
				`{"status":"IN_QUEUE","response_url":"https://queue.fal.run`+falaiTestTaskPath+`"}`,
			),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "missing request ID")
	})

	t.Run("missing response URL", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := falaiFastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			falaiPostSubmitParams(server),
			[]byte(`{"request_id":"req_123","status":"IN_QUEUE"}`),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "missing response URL")
	})

	t.Run("non-JSON poll body", func(t *testing.T) {
		t.Parallel()

		const statusPath = falaiTestTaskPath + "/status"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != statusPath {
				t.Errorf("unexpected request path %q", r.URL.Path)

				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("not json"))
		}))
		defer server.Close()

		codec := falaiFastImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			falaiPostSubmitParams(server),
			[]byte(falaiStateBody("IN_QUEUE", "", "")),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode fal.ai queue response")
		require.Equal(t, int64(1), polls.Load(), "a malformed poll stops polling")
	})

	t.Run("oversized poll body", func(t *testing.T) {
		t.Parallel()

		const statusPath = falaiTestTaskPath + "/status"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != statusPath {
				t.Errorf("unexpected request path %q", r.URL.Path)

				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("IN_PROGRESS", "", "")))
		}))
		defer server.Close()

		codec := falaiFastImageCodec()
		params := falaiPostSubmitParams(server)
		params.MaxResponseBodyBytes = 4
		_, err := codec.PostSubmit(
			context.Background(),
			params,
			[]byte(falaiStateBody("IN_QUEUE", "", "")),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "exceeds max size")
		require.Equal(t, int64(1), polls.Load(), "an oversized poll stops polling")
	})

	t.Run("malformed result bodies", func(t *testing.T) {
		t.Parallel()

		const statusPath = falaiTestTaskPath + "/status"

		cases := []struct {
			name       string
			resultBody string
			postSubmit func(server *httptest.Server) error
			wantErr    string
		}{
			{
				name:       "non-JSON result body",
				resultBody: "not json",
				postSubmit: func(server *httptest.Server) error {
					codec := falaiFastImageCodec()
					_, err := codec.PostSubmit(
						context.Background(),
						falaiPostSubmitParams(server),
						[]byte(falaiStateBody("IN_QUEUE", "", "")),
						http.Header{},
					)

					return err
				},
				wantErr: "failed to decode fal.ai image result",
			},
			{
				name:       "result without images",
				resultBody: `{"images":[]}`,
				postSubmit: func(server *httptest.Server) error {
					codec := falaiFastImageCodec()
					_, err := codec.PostSubmit(
						context.Background(),
						falaiPostSubmitParams(server),
						[]byte(falaiStateBody("IN_QUEUE", "", "")),
						http.Header{},
					)

					return err
				},
				wantErr: "no output URL",
			},
			{
				name:       "result without video",
				resultBody: `{}`,
				postSubmit: func(server *httptest.Server) error {
					codec := falaiFastVideoCodec()
					_, err := codec.PostSubmit(
						context.Background(),
						falaiPostSubmitParams(server),
						[]byte(falaiStateBody("IN_QUEUE", "", "")),
						http.Header{},
					)

					return err
				},
				wantErr: "no output URL",
			},
			{
				name:       "result without segmentation mask",
				resultBody: `{}`,
				postSubmit: func(server *httptest.Server) error {
					codec := falaiFastSegmentationCodec()
					_, err := codec.PostSubmit(
						context.Background(),
						falaiPostSubmitParams(server),
						[]byte(falaiStateBody("IN_QUEUE", "", "")),
						http.Header{},
					)

					return err
				},
				wantErr: "no mask URL",
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				server := httptest.NewServer(
					http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						switch r.URL.Path {
						case statusPath:
							w.Header().Set("Content-Type", "application/json")
							_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
						default:
							w.Header().Set("Content-Type", "application/json")
							_, _ = w.Write([]byte(tc.resultBody))
						}
					}),
				)
				defer server.Close()

				err := tc.postSubmit(server)
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
				require.ErrorContains(t, err, tc.wantErr)
			})
		}
	})
}

// TestFalAIPostSubmit_TextToVideo exercises the text-to-video exchange: the
// completed result's video URL is downloaded and surfaced as video bytes.
func TestFalAIPostSubmit_TextToVideo(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"video":{"url":"` + outputURL + `"}}`))
		default:
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.mp4"

	codec := falaiFastVideoCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'}, resp.Video)
	require.Equal(t, "video/mp4", resp.MediaType)
	require.Equal(t, "req_123", resp.ProviderMetadata["request_id"])
}

// TestFalAIPostSubmit_ImageToImage exercises the image-to-image exchange: the
// completed result's first image URL is downloaded and surfaced.
func TestFalAIPostSubmit_ImageToImage(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageToImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
}

// TestFalAIPostSubmit_Segmentation_DataURL exercises the segmentation
// exchange with an inline data-URL mask: the payload after the comma is used
// as the base64 mask without any download.
func TestFalAIPostSubmit_Segmentation_DataURL(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var downloads atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"image":{"url":"data:image/png;base64,iVBORw0KGgo="}}`))
		default:
			downloads.Add(1)
		}
	}))
	defer server.Close()

	codec := falaiFastSegmentationCodec()
	segments, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Len(t, segments, 1)
	require.Equal(t, "mask", segments[0].Label)
	require.Equal(t, "iVBORw0KGgo=", segments[0].Mask)
	require.Nil(t, segments[0].Score)
	require.Equal(t, "req_123", segments[0].ProviderMetadata["request_id"])
	require.Equal(t, int64(0), downloads.Load(), "an inline mask is never downloaded")
}

// TestFalAIPostSubmit_Segmentation_Download exercises the segmentation
// exchange with a CDN mask URL: the mask is downloaded and re-encoded as
// base64.
func TestFalAIPostSubmit_Segmentation_Download(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"image":{"url":"` + outputURL + `"}}`))
		default:
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/mask.png"

	codec := falaiFastSegmentationCodec()
	segments, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Len(t, segments, 1)
	require.Equal(t, "mask", segments[0].Label)
	require.Equal(t, "iVBORw==", segments[0].Mask)
	require.Equal(t, "req_123", segments[0].ProviderMetadata["request_id"])
}

// TestFalAIPostSubmit_SpeechRecognition exercises the speech-recognition
// exchange: the completed result's transcript and timestamps surface on the
// canonical response.
func TestFalAIPostSubmit_SpeechRecognition(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var downloads atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				`{"text":"hello world","chunks":[{"text":"hello","timestamp":[0.0,1.0]}]}`,
			))
		default:
			downloads.Add(1)
		}
	}))
	defer server.Close()

	codec := falaiFastSpeechCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "hello world", resp.Text)
	require.Equal(t, []hftypes.SpeechRecognitionChunk{
		{Text: "hello", Timestamp: []float64{0.0, 1.0}},
	}, resp.Chunks)
	require.Equal(t, "req_123", resp.ProviderMetadata["request_id"])
	require.Equal(t, int64(0), downloads.Load(), "a transcript never downloads media")
}

// TestFalAIPostSubmit_MediaTypeMismatch exercises the output validation: a
// codec rejects an downloaded output whose media type belongs to another
// family.
func TestFalAIPostSubmit_MediaTypeMismatch(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	cases := []struct {
		name        string
		resultBody  func(outputURL string) string
		outputType  string
		outputBytes []byte
		postSubmit  func(server *httptest.Server) error
	}{
		{
			name:        "image codec gets video",
			resultBody:  func(u string) string { return `{"images":[{"url":"` + u + `"}]}` },
			outputType:  "video/mp4",
			outputBytes: []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			postSubmit: func(server *httptest.Server) error {
				codec := falaiFastImageCodec()
				_, err := codec.PostSubmit(
					context.Background(),
					falaiPostSubmitParams(server),
					[]byte(falaiStateBody("COMPLETED", "", "")),
					http.Header{},
				)

				return err
			},
		},
		{
			name:        "video codec gets image",
			resultBody:  func(u string) string { return `{"video":{"url":"` + u + `"}}` },
			outputType:  "image/png",
			outputBytes: []byte{0x89, 'P', 'N', 'G'},
			postSubmit: func(server *httptest.Server) error {
				codec := falaiFastVideoCodec()
				_, err := codec.PostSubmit(
					context.Background(),
					falaiPostSubmitParams(server),
					[]byte(falaiStateBody("COMPLETED", "", "")),
					http.Header{},
				)

				return err
			},
		},
		{
			name:        "segmentation codec gets video",
			resultBody:  func(u string) string { return `{"image":{"url":"` + u + `"}}` },
			outputType:  "video/mp4",
			outputBytes: []byte{0x00, 0x00, 0x00, 0x20, 'f', 't', 'y', 'p'},
			postSubmit: func(server *httptest.Server) error {
				codec := falaiFastSegmentationCodec()
				_, err := codec.PostSubmit(
					context.Background(),
					falaiPostSubmitParams(server),
					[]byte(falaiStateBody("COMPLETED", "", "")),
					http.Header{},
				)

				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var outputURL string
			server := httptest.NewServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case statusPath:
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
					case falaiTestTaskPath:
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(tc.resultBody(outputURL)))
					default:
						w.Header().Set("Content-Type", tc.outputType)
						_, _ = w.Write(tc.outputBytes)
					}
				}),
			)
			defer server.Close()
			outputURL = server.URL + "/cdn/output"

			err := tc.postSubmit(server)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		})
	}
}

// TestFalAIPostSubmit_SniffedMediaType exercises the media-type fallback:
// when the CDN omits Content-Type, the bytes are sniffed.
func TestFalAIPostSubmit_SniffedMediaType(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("COMPLETED", "", "")),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
}

// TestFalAIPostSubmit_ContextCanceled exercises cancellation: a canceled
// context aborts the exchange with the context error.
func TestFalAIPostSubmit_ContextCanceled(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		ctx,
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("IN_QUEUE", "", "")),
		http.Header{},
	)
	require.ErrorIs(t, err, context.Canceled)
}

// TestFalAIPostSubmit_ContextCanceledDuringPoll exercises cancellation after
// polling has started: the context is canceled while the second poll request
// is in flight, and the resulting transport error — which wraps
// context.Canceled — must surface immediately instead of being swallowed as
// a transient failure worth retrying.
func TestFalAIPostSubmit_ContextCanceledDuringPoll(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var polls atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// pollStarted is closed when the second poll arrives; release unblocks
	// its handler so the test can cancel the context mid-roundtrip.
	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != statusPath {
			t.Errorf("unexpected request path %q", r.URL.Path)

			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(falaiStateBody("IN_PROGRESS", "", "")))
	}))
	defer server.Close()

	type submitted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan submitted, 1)
	codec := falaiFastImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			falaiPostSubmitParams(server),
			[]byte(falaiStateBody("IN_QUEUE", "", "")),
			http.Header{},
		)
		results <- submitted{resp: resp, err: err}
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

// TestFalAIPostSubmit_DeadlineExceededDuringPoll exercises deadline expiry
// after polling has started: the caller's deadline lapses while a poll
// request is in flight, and the resulting transport error — which wraps
// context.DeadlineExceeded — must surface immediately instead of being
// swallowed as a transient failure worth retrying.
func TestFalAIPostSubmit_DeadlineExceededDuringPoll(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var polls atomic.Int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != statusPath {
			t.Errorf("unexpected request path %q", r.URL.Path)

			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(falaiStateBody("IN_PROGRESS", "", "")))
	}))
	defer server.Close()

	type submitted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan submitted, 1)
	codec := falaiFastImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			falaiPostSubmitParams(server),
			[]byte(falaiStateBody("IN_QUEUE", "", "")),
			http.Header{},
		)
		results <- submitted{resp: resp, err: err}
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

// TestFalAIQueueBase exercises the follow-up base resolution: the Hugging
// Face router gains the /fal-ai prefix, other hosts resolve as-is, and an
// unparseable base URL is a configuration error.
func TestFalAIQueueBase(t *testing.T) {
	t.Parallel()

	t.Run("router host gains /fal-ai prefix", func(t *testing.T) {
		t.Parallel()

		base, routed, err := falaiPollBase("https://router.huggingface.co")
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://router.huggingface.co/fal-ai", base)
	})

	t.Run("router base with trailing slash", func(t *testing.T) {
		t.Parallel()

		base, routed, err := falaiPollBase("https://router.huggingface.co/")
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://router.huggingface.co/fal-ai", base)
	})

	t.Run("other hosts resolve as-is", func(t *testing.T) {
		t.Parallel()

		base, routed, err := falaiPollBase("https://queue.fal.run")
		require.NoError(t, err)
		require.False(t, routed)
		require.Equal(t, "https://queue.fal.run", base)
	})

	t.Run("invalid base URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := falaiPollBase("http://[::1:bad")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

// TestFalAIResultPath exercises the response-URL path extraction: absolute
// URLs contribute only their path (so a foreign host cannot redirect the
// poll), relative paths gain a leading slash, and unparseable URLs are
// serialization errors.
func TestFalAIResultPath(t *testing.T) {
	t.Parallel()

	t.Run("absolute URL resolves from its path", func(t *testing.T) {
		t.Parallel()

		path, err := falaiResultPath("https://queue.fal.run/fal-ai/flux/requests/req_1")
		require.NoError(t, err)
		require.Equal(t, "/fal-ai/flux/requests/req_1", path)
	})

	t.Run("relative path gains a leading slash", func(t *testing.T) {
		t.Parallel()

		path, err := falaiResultPath("fal-ai/flux/requests/req_1")
		require.NoError(t, err)
		require.Equal(t, "/fal-ai/flux/requests/req_1", path)
	})

	t.Run("empty path normalizes to root", func(t *testing.T) {
		t.Parallel()

		path, err := falaiResultPath("https://queue.fal.run")
		require.NoError(t, err)
		// The exchange then fails at the status request, which is the
		// provider's concern, not the extractor's.
		require.Equal(t, "/", path)
	})

	t.Run("unparseable URL is an error", func(t *testing.T) {
		t.Parallel()

		_, err := falaiResultPath("http://h/%zz")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestFalAIPostSubmit_ForeignResponseURLCannotRedirect pins the poll
// invariant end to end: a provider-supplied absolute response URL pointing
// at a foreign host cannot redirect the token-bearing poll — the host is
// discarded and the path resolves against the caller's base URL.
func TestFalAIPostSubmit_ForeignResponseURLCannotRedirect(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	foreign := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("a provider-supplied absolute response URL must not redirect the poll")
	}))
	defer foreign.Close()

	var (
		polls         atomic.Int64
		downloads     atomic.Int64
		outputURL     string
		gotPollAuth   string
		gotResultAuth string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			gotPollAuth = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write([]byte(falaiStateBody("IN_PROGRESS", "", "")))

				return
			}
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			gotResultAuth = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		// The submit response carries an absolute response URL pointing at a
		// foreign host; the poll must still resolve to the caller's host.
		[]byte(falaiState("IN_QUEUE", map[string]any{
			"response_url": foreign.URL + falaiTestTaskPath,
		})),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())
	// The token stays on the caller's base host, never the foreign host.
	require.Equal(t, "Bearer test-token", gotPollAuth)
	require.Equal(t, "Bearer test-token", gotResultAuth)
}

// TestFalAIPostSubmit_InvalidOutputURL pins the structural credential guard:
// an output URL that is not an absolute http(s) URL is a configuration
// error, and no request is made for it.
func TestFalAIPostSubmit_InvalidOutputURL(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"cdn/output.png"}]}`))
		default:
			t.Errorf("no request should be made for an invalid output URL, got %q", r.URL.Path)
		}
	}))
	defer server.Close()

	codec := falaiFastImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		falaiPostSubmitParams(server),
		[]byte(falaiStateBody("COMPLETED", "", "")),
		http.Header{},
	)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

// TestFalAIPostSubmit_DownloadBodyLimit exercises the download-side
// response-size cap: a completed task whose CDN output exceeds
// PostSubmitParams.MaxResponseBodyBytes fails instead of downloading without
// bound.
func TestFalAIPostSubmit_DownloadBodyLimit(t *testing.T) {
	t.Parallel()

	const statusPath = falaiTestTaskPath + "/status"

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case statusPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(falaiStateBody("COMPLETED", "", "")))
		case falaiTestTaskPath:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"images":[{"url":"` + outputURL + `"}]}`))
		default:
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', 0, 0, 0, 0, 0, 0, 0, 0})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := falaiFastImageCodec()
	params := falaiPostSubmitParams(server)
	params.MaxResponseBodyBytes = 4
	_, err := codec.PostSubmit(
		context.Background(),
		params,
		[]byte(falaiStateBody("COMPLETED", "", "")),
		http.Header{},
	)
	require.Error(t, err)
}

// TestFalAIRetryAfter exercises the Retry-After parser as a table: seconds
// are honored, garbage and non-positive values are ignored, and the cap
// bounds pathological values.
func TestFalAIRetryAfter(t *testing.T) {
	t.Parallel()

	cases := []struct {
		header string
		want   time.Duration
	}{
		{header: "3", want: 3 * time.Second},
		{header: "soon", want: 0},
		{header: "0", want: 0},
		{header: "-3", want: 0},
		{header: "999999", want: falaiMaxRetryAfter},
	}

	for _, tc := range cases {
		headers := http.Header{}
		headers.Set("Retry-After", tc.header)

		require.Equal(t, tc.want, falaiRetryAfter(headers), "Retry-After %q", tc.header)
	}
}

// TestFalAIMetadata exercises provider-metadata extraction: an empty task
// state yields nil so callers never observe an empty map.
func TestFalAIMetadata(t *testing.T) {
	t.Parallel()

	t.Run("empty task state", func(t *testing.T) {
		t.Parallel()

		require.Nil(t, falaiMetadata(&falaiTaskState{}))
	})

	t.Run("populated task state", func(t *testing.T) {
		t.Parallel()

		position := 3
		inference := 2.5
		meta := falaiMetadata(&falaiTaskState{
			RequestID:     "req_1",
			QueuePosition: &position,
			Metrics:       &falaiMetrics{InferenceTime: &inference},
		})
		require.Equal(t, "req_1", meta["request_id"])
		require.Equal(t, 3, meta["queue_position"])
		require.InEpsilon(t, 2.5, meta["inference_time"], 0.001)
	})
}

// TestFalAITaskFailedError exercises the terminal-failure error message: the
// provider's error message, category, and request ID are appended only when
// present.
func TestFalAITaskFailedError(t *testing.T) {
	t.Parallel()

	errMsg := "CUDA out of memory"
	errType := "runner_error"
	reqID := "req_1"

	cases := []struct {
		name  string
		state falaiTaskState
		want  string
	}{
		{
			name:  "bare failure",
			state: falaiTaskState{Status: "COMPLETED"},
			want:  "fal.ai task failed",
		},
		{
			name:  "with provider error",
			state: falaiTaskState{Status: "COMPLETED", Error: &errMsg},
			want:  "fal.ai task failed: CUDA out of memory",
		},
		{
			name:  "with error type",
			state: falaiTaskState{Status: "COMPLETED", ErrorType: &errType},
			want:  "fal.ai task failed [runner_error]",
		},
		{
			name:  "with request id",
			state: falaiTaskState{Status: "COMPLETED", RequestID: reqID},
			want:  "fal.ai task failed (request id req_1)",
		},
		{
			name: "with everything",
			state: falaiTaskState{
				Status:    "COMPLETED",
				RequestID: reqID,
				Error:     &errMsg,
				ErrorType: &errType,
			},
			want: "fal.ai task failed: CUDA out of memory [runner_error] (request id req_1)",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			state := tc.state
			err := falaiTaskFailedError(&state)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

// TestFalAIFlattenParameters exercises the parameter-flattening helper: nil
// parameters flatten to an empty map, marshal and flatten failures surface
// as serialization SDK errors, and valid parameters flatten to their JSON
// object form.
func TestFalAIFlattenParameters(t *testing.T) {
	t.Parallel()

	flat, err := falaiFlattenParameters(nil)
	require.NoError(t, err)
	require.Empty(t, flat)

	_, err = falaiFlattenParameters(failingParameters{})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)

	_, err = falaiFlattenParameters(arrayParameters{})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)

	flat, err = falaiFlattenParameters(hftypes.TextToVideoParameters{Seed: new(int64(42))})
	require.NoError(t, err)
	require.Equal(t, map[string]any{"seed": float64(42)}, flat)
}

// TestFalAINormalizeDataInput exercises the data-URI wrapping: data URIs and
// HTTP(S) URLs pass through, raw base64 wraps with the given media type, and
// an empty input passes through unchanged.
func TestFalAINormalizeDataInput(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		input      string
		mediaType  string
		wantOutput string
	}{
		{
			name:       "data URI passes through",
			input:      "data:image/png;base64,abc",
			mediaType:  "image/jpeg",
			wantOutput: "data:image/png;base64,abc",
		},
		{
			name:       "http URL passes through",
			input:      "http://example.com/a.png",
			mediaType:  "image/jpeg",
			wantOutput: "http://example.com/a.png",
		},
		{
			name:       "https URL passes through",
			input:      "https://example.com/a.png",
			mediaType:  "image/jpeg",
			wantOutput: "https://example.com/a.png",
		},
		{
			name:       "raw base64 wraps",
			input:      "abc",
			mediaType:  "audio/mpeg",
			wantOutput: "data:audio/mpeg;base64,abc",
		},
		{
			name:       "empty input passes through",
			input:      "",
			mediaType:  "image/jpeg",
			wantOutput: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.wantOutput, falaiNormalizeDataInput(tc.input, tc.mediaType))
		})
	}
}

// TestFalAIApplyImageSize exercises the width/height fold: a complete pair
// becomes an image_size object, a lone dimension stays flat.
func TestFalAIApplyImageSize(t *testing.T) {
	t.Parallel()

	t.Run("complete pair folds", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"width": 512, "height": 768, "seed": 1}
		falaiApplyImageSize(payload)
		require.Equal(t, map[string]any{
			"image_size": map[string]any{"width": 512, "height": 768},
			"seed":       1,
		}, payload)
	})

	t.Run("lone width stays", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"width": 512}
		falaiApplyImageSize(payload)
		require.Equal(t, map[string]any{"width": 512}, payload)
	})

	t.Run("lone height stays", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"height": 768}
		falaiApplyImageSize(payload)
		require.Equal(t, map[string]any{"height": 768}, payload)
	})
}

// TestFalAIApplyTargetSize exercises the image-to-image size rename:
// target_size becomes image_size, and a payload without target_size is
// untouched.
func TestFalAIApplyTargetSize(t *testing.T) {
	t.Parallel()

	t.Run("target size renames", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{
			"target_size": map[string]any{"width": 256, "height": 256},
			"prompt":      "a cat",
		}
		falaiApplyTargetSize(payload)
		require.Equal(t, map[string]any{
			"image_size": map[string]any{"width": 256, "height": 256},
			"prompt":     "a cat",
		}, payload)
	})

	t.Run("without target size untouched", func(t *testing.T) {
		t.Parallel()

		payload := map[string]any{"prompt": "a cat"}
		falaiApplyTargetSize(payload)
		require.Equal(t, map[string]any{"prompt": "a cat"}, payload)
	})
}

// TestFalAIInlineDataURL exercises the inline-mask shortcut: data URLs with
// a payload return their base64 payload as-is, while anything else — a data
// URL without a comma, an http(s) URL, or a bare path — reports not-inline
// so the caller downloads it.
func TestFalAIInlineDataURL(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		ref         string
		wantPayload string
		wantOK      bool
	}{
		{
			name:        "base64 data URL",
			ref:         "data:image/png;base64,iVBORw0KGgo=",
			wantPayload: "iVBORw0KGgo=",
			wantOK:      true,
		},
		{
			name:   "data URL without a comma is not inline",
			ref:    "data:image/png",
			wantOK: false,
		},
		{
			name:   "http URL is not inline",
			ref:    "https://cdn.example.com/mask.png",
			wantOK: false,
		},
		{
			name:   "bare path is not inline",
			ref:    "cdn/mask.png",
			wantOK: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			payload, ok := falaiInlineDataURL(tc.ref)
			require.Equal(t, tc.wantOK, ok)
			require.Equal(t, tc.wantPayload, payload)
		})
	}
}

// TestFalAICodec_EncodeUnmarshalableProviderParams exercises every task
// codec's encode-side marshal guard: a ProviderParams entry json cannot
// encode makes the whole flat body unmarshalable, which surfaces as a
// serialization error instead of a garbled request.
func TestFalAICodec_EncodeUnmarshalableProviderParams(t *testing.T) {
	t.Parallel()

	t.Run("text-to-image", func(t *testing.T) {
		t.Parallel()

		codec := falaiTextToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "fal-ai/krea-2/turbo",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("text-to-video", func(t *testing.T) {
		t.Parallel()

		codec := falaiTextToVideoCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "fal-ai/wan/v2.2-ti2v-5b/text-to-video",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("image-to-image", func(t *testing.T) {
		t.Parallel()

		codec := falaiImageToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{
				Input:          "data:image/png;base64,abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "fal-ai/flux-2/klein/9b/edit",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("image-segmentation", func(t *testing.T) {
		t.Parallel()

		codec := falaiImageSegmentationCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.ImageSegmentationRequest]{
			Context: context.Background(),
			Request: hftypes.ImageSegmentationRequest{
				Input:          "abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "fal-ai/bria/background/remove",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})

	t.Run("speech-recognition", func(t *testing.T) {
		t.Parallel()

		codec := falaiSpeechRecognitionCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          "abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "fal-ai/whisper",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to marshal request body")
	})
}
