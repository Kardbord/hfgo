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
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const (
	// base64JpegMagic is the base64 encoding of the JPEG file signature
	// (0xFF, 0xD8, 0xFF, 0xE0), exercising the input codecs' byte sniffing.
	base64JpegMagic = "/9j/4AA="
	// base64PngMagic is the base64 encoding of the PNG file signature
	// (0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a).
	base64PngMagic = "iVBORw0KGgo="
)

func TestReplicateProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewReplicateProvider(mapper)
	require.NoError(t, err)

	require.Equal(t, "replicate", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

func TestNewReplicateProvider_NilResolver(t *testing.T) {
	t.Parallel()

	p, err := NewReplicateProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, ReplicateProvider{}, p)
}

func TestReplicateProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var p ReplicateProvider

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

	_, err = p.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// replicateEndpointCase pairs a display name with the endpoint method it
// exercises, the Replicate mapping the method must resolve to, and the
// router-proxied prediction path the endpoint must carry.
type replicateEndpointCase struct {
	name            string
	task            Task
	model           string
	providerModelID string
	wantPath        string
	endpoint        func(ReplicateProvider, EndpointParams) (Endpoint, error)
}

func TestReplicateProvider_Endpoints(t *testing.T) {
	t.Parallel()

	cases := []replicateEndpointCase{
		{
			name:            "TextToVideoEndpoint",
			task:            TaskTextToVideo,
			model:           "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
			providerModelID: "wan-video/wan-2.1-t2v-14b",
			wantPath:        "replicate/v1/models/wan-video/wan-2.1-t2v-14b/predictions",
			endpoint:        ReplicateProvider.TextToVideoEndpoint,
		},
		{
			name:            "TextToImageEndpoint",
			task:            TaskTextToImage,
			model:           "black-forest-labs/FLUX.1-dev",
			providerModelID: "black-forest-labs/flux-dev",
			wantPath:        "replicate/v1/models/black-forest-labs/flux-dev/predictions",
			endpoint:        ReplicateProvider.TextToImageEndpoint,
		},
		{
			name:            "ImageToImageEndpoint",
			task:            TaskImageToImage,
			model:           "black-forest-labs/FLUX.2-klein-base-4B",
			providerModelID: "bria/fibo-2.0/edit",
			wantPath:        "replicate/v1/models/bria/fibo-2.0/edit/predictions",
			endpoint:        ReplicateProvider.ImageToImageEndpoint,
		},
		{
			name:            "SpeechRecognitionEndpoint",
			task:            TaskAutomaticSpeechRecognition,
			model:           "openai/whisper-large-v3",
			providerModelID: "openai/whisper-large-v3",
			wantPath:        "replicate/v1/models/openai/whisper-large-v3/predictions",
			endpoint:        ReplicateProvider.SpeechRecognitionEndpoint,
		},
		{
			name:            "version pin uses the versionless route",
			task:            TaskTextToImage,
			model:           "black-forest-labs/FLUX.1-dev",
			providerModelID: "black-forest-labs/flux-dev:abc123def",
			wantPath:        "replicate/v1/predictions",
			endpoint:        ReplicateProvider.TextToImageEndpoint,
		},
	}
	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mapper := NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: replicateProviderName,
					Task:     tc.task,
					Model:    tc.model,
				}: {ProviderModelID: tc.providerModelID},
			})
			p, err := NewReplicateProvider(mapper)
			require.NoError(t, err)

			endpoint, err := tc.endpoint(p, EndpointParams{
				Context: context.Background(),
				Model:   tc.model,
			})
			require.NoError(t, err)
			require.Equal(t, http.MethodPost, endpoint.Method)
			require.Equal(t, tc.wantPath, endpoint.Path)
			require.NotNil(t, endpoint.Mapping)
			require.Equal(t, tc.providerModelID, endpoint.Mapping.ProviderModelID)
		})
	}
}

func TestReplicateProvider_Endpoint_ResolutionError(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewReplicateProvider(mapper)
	require.NoError(t, err)

	_, err = p.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "unknown-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// requireReplicateEncode asserts an encoder produced the expected
// prediction payload with the headers Replicate submissions carry.
func requireReplicateEncode[Req any](
	t *testing.T,
	encode func(EncodeParams[Req]) ([]byte, http.Header, error),
	req Req,
	model string,
	mapping *ModelMapping,
	wantJSON string,
) {
	t.Helper()

	body, headers, err := encode(EncodeParams[Req]{
		Context: context.Background(),
		Request: req,
		Model:   model,
		Mapping: mapping,
	})
	require.NoError(t, err)
	require.JSONEq(t, wantJSON, string(body))
	//nolint:testifylint // media type
	require.Equal(t, mimeTypeJSON, headers.Get("Content-Type"))
	//nolint:testifylint // media type
	require.Equal(t, mimeTypeJSON, headers.Get(headerAccept))
	require.Equal(t, replicatePreferWait, headers.Get(replicatePreferKey))
}

func TestReplicateProvider_TextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := ReplicateTextToImageMixin{}.TextToImageCodec()

	t.Run("prompt only", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			"black-forest-labs/FLUX.1-dev",
			&ModelMapping{ProviderModelID: "black-forest-labs/flux-dev"},
			`{"input":{"prompt":"a cat"}}`,
		)
	})

	t.Run("parameters merged into input", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					GuidanceScale:     new(7.5),
					NumInferenceSteps: new(30),
				},
			},
			"black-forest-labs/FLUX.1-dev",
			&ModelMapping{ProviderModelID: "black-forest-labs/flux-dev"},
			`{"input":{"prompt":"a cat","guidance_scale":7.5,"num_inference_steps":30}}`,
		)
	})

	t.Run("provider params override typed fields", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					NumInferenceSteps: new(30),
				},
				ProviderParams: map[string]any{
					"num_inference_steps":   48,
					"enable_safety_checker": true,
				},
			},
			"black-forest-labs/FLUX.1-dev",
			&ModelMapping{ProviderModelID: "black-forest-labs/flux-dev"},
			`{"input":{"prompt":"a cat","num_inference_steps":48,`+
				`"enable_safety_checker":true}}`,
		)
	})

	t.Run("version pinned payload", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			"black-forest-labs/FLUX.1-dev",
			&ModelMapping{ProviderModelID: "black-forest-labs/flux-dev:abc123def"},
			`{"input":{"prompt":"a cat"},"version":"abc123def"}`,
		)
	})

	t.Run("no mapping encodes without adapter handling", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToImageCodec{}
		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			"black-forest-labs/FLUX.1-dev",
			nil,
			`{"input":{"prompt":"a cat"}}`,
		)
	})
}

func TestReplicateProvider_TextToVideoCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := ReplicateTextToVideoMixin{}.TextToVideoCodec()

	t.Run("prompt only", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{Input: "a cat"},
			"Wan-AI/Wan2.2-T2V-A14B-Diffusers",
			&ModelMapping{ProviderModelID: "wan-video/wan-2.1-t2v-14b"},
			`{"input":{"prompt":"a cat"}}`,
		)
	})

	t.Run("parameters merged into input", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToVideoRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToVideoParameters{
					NumFrames: new(24),
					Seed:      new(int64(42)),
				},
			},
			"Wan-AI/Wan2.2-T2V-A14B-Diffusers",
			&ModelMapping{ProviderModelID: "wan-video/wan-2.1-t2v-14b"},
			`{"input":{"prompt":"a cat","num_frames":24,"seed":42}}`,
		)
	})

	t.Run("resolution failure surfaces as an error", func(t *testing.T) {
		t.Parallel()

		codec := ReplicateTextToVideoMixin{Resolver: sdkFailingResolver{}}.TextToVideoCodec()
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{Input: "a cat"},
			Model:   "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

func TestReplicateProvider_ImageToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := ReplicateImageToImageMixin{}.ImageToImageCodec()

	t.Run("data URI rides under all four image keys", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
				Parameters: &hftypes.ImageToImageParameters{
					Prompt: new("make it blue"),
				},
			},
			"black-forest-labs/FLUX.2-klein-base-4B",
			&ModelMapping{ProviderModelID: "bria/fibo-2.0/edit"},
			`{"input":{`+
				`"image":"data:image/png;base64,abc123",`+
				`"images":["data:image/png;base64,abc123"],`+
				`"input_image":"data:image/png;base64,abc123",`+
				`"input_images":["data:image/png;base64,abc123"],`+
				`"prompt":"make it blue"}}`,
		)
	})

	t.Run("http URL rides under all four image keys", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: "https://huggingface.co/image.png",
			},
			"black-forest-labs/FLUX.2-klein-base-4B",
			&ModelMapping{ProviderModelID: "bria/fibo-2.0/edit"},
			`{"input":{`+
				`"image":"https://huggingface.co/image.png",`+
				`"images":["https://huggingface.co/image.png"],`+
				`"input_image":"https://huggingface.co/image.png",`+
				`"input_images":["https://huggingface.co/image.png"]}}`,
		)
	})

	t.Run("raw base64 jpeg labeled by sniffing", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: base64JpegMagic,
			},
			"black-forest-labs/FLUX.2-klein-base-4B",
			&ModelMapping{ProviderModelID: "bria/fibo-2.0/edit"},
			`{"input":{`+
				`"image":"data:image/jpeg;base64,`+base64JpegMagic+`",`+
				`"images":["data:image/jpeg;base64,`+base64JpegMagic+`"],`+
				`"input_image":"data:image/jpeg;base64,`+base64JpegMagic+`",`+
				`"input_images":["data:image/jpeg;base64,`+base64JpegMagic+`"]}}`,
		)
	})

	t.Run("raw png bytes labeled by sniffing", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.ImageToImageRequest{
				Input: base64PngMagic,
			},
			"black-forest-labs/FLUX.2-klein-base-4B",
			&ModelMapping{ProviderModelID: "bria/fibo-2.0/edit"},
			`{"input":{`+
				`"image":"data:image/png;base64,`+base64PngMagic+`",`+
				`"images":["data:image/png;base64,`+base64PngMagic+`"],`+
				`"input_image":"data:image/png;base64,`+base64PngMagic+`",`+
				`"input_images":["data:image/png;base64,`+base64PngMagic+`"]}}`,
		)
	})

	t.Run("invalid base64 input is a serialization error", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{Input: "not-base64!!!"},
			Model:   "black-forest-labs/FLUX.2-klein-base-4B",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode base64 input")
	})

	t.Run("resolution failure surfaces as an error", func(t *testing.T) {
		t.Parallel()

		codec := ReplicateImageToImageMixin{Resolver: sdkFailingResolver{}}.ImageToImageCodec()
		_, _, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{Input: "data:image/png;base64,abc123"},
			Model:   "black-forest-labs/FLUX.2-klein-base-4B",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

func TestReplicateProvider_SpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := ReplicateSpeechRecognitionMixin{}.SpeechRecognitionCodec()

	t.Run("http URL rides under audio key", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "https://huggingface.co/audio.wav",
			},
			"openai/whisper-large-v3",
			&ModelMapping{ProviderModelID: "openai/whisper-large-v3"},
			`{"input":{"audio":"https://huggingface.co/audio.wav"}}`,
		)
	})

	t.Run("data URI rides under audio key", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "data:audio/mp3;base64,abc123",
			},
			"openai/whisper-large-v3",
			&ModelMapping{ProviderModelID: "openai/whisper-large-v3"},
			`{"input":{"audio":"data:audio/mp3;base64,abc123"}}`,
		)
	})

	t.Run("raw base64 labeled audio/wav", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "abcdefgh",
			},
			"openai/whisper-large-v3",
			&ModelMapping{ProviderModelID: "openai/whisper-large-v3"},
			`{"input":{"audio":"data:audio/wav;base64,abcdefgh"}}`,
		)
	})

	t.Run("flac bytes labeled by sniffing", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "ZkxhQw==", // "fLaC"
			},
			"openai/whisper-large-v3",
			&ModelMapping{ProviderModelID: "openai/whisper-large-v3"},
			`{"input":{"audio":"data:audio/flac;base64,ZkxhQw=="}}`,
		)
	})

	t.Run("return_timestamps parameter", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.SpeechRecognitionRequest{
				Input: "data:audio/mp3;base64,abc123",
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
			},
			"openai/whisper-large-v3",
			&ModelMapping{ProviderModelID: "openai/whisper-large-v3"},
			`{"input":{"audio":"data:audio/mp3;base64,abc123","return_timestamps":true}}`,
		)
	})

	t.Run("invalid base64 errors", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: "not base64!!",
			},
			Model: "openai/whisper-large-v3",
			Mapping: &ModelMapping{
				ProviderModelID: "openai/whisper-large-v3",
			},
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode base64 input")
	})

	t.Run("resolution failure surfaces as an error", func(t *testing.T) {
		t.Parallel()

		codec := ReplicateSpeechRecognitionMixin{
			Resolver: sdkFailingResolver{},
		}.SpeechRecognitionCodec()
		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: "data:audio/mp3;base64,abc123",
			},
			Model: "openai/whisper-large-v3",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

// fastReplicateImageCodec returns a text-to-image codec with fast polling,
// so tests exercise the exchange's branching without real delays.
func fastReplicateImageCodec() replicateTextToImageCodec {
	return replicateTextToImageCodec{replicateAsyncCodec{queueTaskCodec{
		mediaPrefix:     replicateImageMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// fastReplicateVideoCodec mirrors fastReplicateImageCodec for the
// text-to-video prefix.
func fastReplicateVideoCodec() replicateTextToVideoCodec {
	return replicateTextToVideoCodec{replicateAsyncCodec{queueTaskCodec{
		mediaPrefix:     replicateVideoMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// fastReplicateSpeechCodec mirrors fastReplicateImageCodec for speech
// recognition.
func fastReplicateSpeechCodec() replicateSpeechRecognitionCodec {
	return replicateSpeechRecognitionCodec{replicateAsyncCodec{queueTaskCodec{
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// fastReplicateImageToImageCodec mirrors fastReplicateImageCodec for the
// image-to-image codec, whose exchange is otherwise identical.
func fastReplicateImageToImageCodec() replicateImageToImageCodec {
	return replicateImageToImageCodec{replicateAsyncCodec{queueTaskCodec{
		mediaPrefix:     replicateImageMediaPrefix,
		pollInterval:    time.Millisecond,
		pollStep:        time.Millisecond,
		maxPollInterval: 2 * time.Millisecond,
	}}}
}

// replicatePredictionBody builds a Replicate prediction response body for the
// given data fields. It marshals the response through encoding/json rather
// than string concatenation, so any error message or URL is escaped
// correctly. A nil output marshals as JSON null, the value uncompleted
// predictions carry. Each variadic map merges extra top-level fields (e.g.
// created_at, metrics) over the base fields, in order.
func replicatePredictionBody(
	status, getURL, predictionErr string, output any, extras ...map[string]any,
) string {
	pred := map[string]any{
		"id":     "pred_123",
		"model":  "black-forest-labs/flux-dev",
		"status": status,
		"output": output,
	}
	if getURL != "" {
		pred["urls"] = map[string]any{"get": getURL}
	}
	if predictionErr != "" {
		pred["error"] = predictionErr
	}
	for _, extra := range extras {
		maps.Copy(pred, extra)
	}

	body, err := json.Marshal(pred)
	if err != nil {
		// Unreachable: every value above marshals without error.
		panic(err)
	}

	return string(body)
}

func TestReplicateProvider_LoraInjected(t *testing.T) {
	t.Parallel()

	resolver := NewStaticModelResolver(map[StaticMapping]ModelMapping{
		{
			Provider: replicateProviderName,
			Task:     TaskTextToImage,
			Model:    "some-lora/flux-lora",
		}: {
			ProviderModelID:    "black-forest-labs/flux-dev",
			Adapter:            adapterLora,
			AdapterWeightsPath: "some-lora/flux-lora",
		},
		{
			Provider: replicateProviderName,
			Task:     TaskImageToImage,
			Model:    "some-lora/flux-lora",
		}: {
			ProviderModelID:    "bria/fibo-2.0/edit",
			Adapter:            adapterLora,
			AdapterWeightsPath: "some-lora/flux-lora",
		},
	})
	imageCodec := ReplicateTextToImageMixin{Resolver: resolver}.TextToImageCodec()
	imageToImageCodec := ReplicateImageToImageMixin{Resolver: resolver}.ImageToImageCodec()

	t.Run("text-to-image", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			imageCodec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			"some-lora/flux-lora",
			nil,
			`{"input":{"prompt":"a cat",`+
				`"lora_weights":"https://huggingface.co/some-lora/flux-lora"}}`,
		)
	})

	t.Run("image-to-image", func(t *testing.T) {
		t.Parallel()

		requireReplicateEncode(
			t,
			imageToImageCodec.Encode,
			hftypes.ImageToImageRequest{
				Input: "data:image/png;base64,abc123",
			},
			"some-lora/flux-lora",
			nil,
			`{"input":{`+
				`"image":"data:image/png;base64,abc123",`+
				`"images":["data:image/png;base64,abc123"],`+
				`"input_image":"data:image/png;base64,abc123",`+
				`"input_images":["data:image/png;base64,abc123"],`+
				`"lora_weights":"https://huggingface.co/some-lora/flux-lora"}}`,
		)
	})
}

func TestReplicateProvider_LoraNotInjected(t *testing.T) {
	t.Parallel()

	t.Run("no adapter", func(t *testing.T) {
		t.Parallel()

		codec := ReplicateTextToImageMixin{}.TextToImageCodec()
		requireReplicateEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			"black-forest-labs/FLUX.1-dev",
			nil,
			`{"input":{"prompt":"a cat"}}`,
		)
	})

	t.Run("empty adapter weights", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToImageCodec{}
		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "some-lora/flux-lora",
			Mapping: &ModelMapping{
				ProviderModelID: "black-forest-labs/flux-dev",
				Adapter:         adapterLora,
			},
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"input":{"prompt":"a cat"}}`, string(body))
	})

	t.Run("caller-supplied lora_weights win", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToImageCodec{}
		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					replicateLoraWeightsKey: "https://huggingface.co/user/own-lora",
				},
			},
			Model: "some-lora/flux-lora",
			Mapping: &ModelMapping{
				ProviderModelID:    "black-forest-labs/flux-dev",
				Adapter:            adapterLora,
				AdapterWeightsPath: "some-lora/flux-lora",
			},
		})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"input":{"prompt":"a cat","lora_weights":"https://huggingface.co/user/own-lora"}}`,
			string(body),
		)
	})

	t.Run("lookup failure propagates", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToImageCodec{
			replicateAsyncCodec{queueTaskCodec{resolver: &sdkFailingResolver{}}},
		}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   "unknown-model",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "lookup failure")
	})
}

func TestReplicateProvider_TextToImageCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := ReplicateTextToImageMixin{}.TextToImageCodec()
	assertImageMediaDecode(t, codec.Decode, func(r hftypes.TextToImageResponse) ([]byte, string) {
		return r.Image, r.MediaType
	})
}

func TestReplicateProvider_ImageToImageCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := ReplicateImageToImageMixin{}.ImageToImageCodec()
	assertImageMediaDecode(t, codec.Decode, func(r hftypes.ImageToImageResponse) ([]byte, string) {
		return r.Image, r.MediaType
	})
}

func TestReplicateProvider_TextToVideoCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := ReplicateTextToVideoMixin{}.TextToVideoCodec()

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

func TestReplicateProvider_SpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := ReplicateSpeechRecognitionMixin{}.SpeechRecognitionCodec()

	t.Run("valid transcript", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"hello world"}`),
			Headers: headers,
		})
		require.NoError(t, err)
		require.Equal(t, "hello world", resp.Text)
	})

	t.Run("invalid content type", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "image/png")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"hello"}`),
			Headers: headers,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("malformed body", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte("not json"),
			Headers: headers,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestReplicatePostSubmit_Complete exercises the exchange when the submit
// response already carries the completed prediction (sync mode): the output
// is downloaded without any polling requests.
func TestReplicatePostSubmit_Complete(t *testing.T) {
	t.Parallel()

	var (
		polls         atomic.Int64
		downloads     atomic.Int64
		gotDownloadUA string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/predictions/pred_123" {
			polls.Add(1)

			return
		}
		downloads.Add(1)
		gotDownloadUA = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody(
			"succeeded", "", "", server.URL+"/output.png",
			map[string]any{"created_at": "2026-01-01T00:00:00Z"},
			map[string]any{"metrics": map[string]any{"predict_time": 1.5}},
		)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(0), polls.Load(), "a completed submit response needs no polling")
	require.Equal(t, int64(1), downloads.Load())
	require.Equal(t, "test-agent/1.0", gotDownloadUA)
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
	require.Equal(t, "black-forest-labs/flux-dev", resp.ProviderMetadata["model"])
	require.Equal(t, "succeeded", resp.ProviderMetadata["status"])
	require.Equal(t, "2026-01-01T00:00:00Z", resp.ProviderMetadata["created_at"])
	require.Equal(t, map[string]any{"predict_time": 1.5}, resp.ProviderMetadata["metrics"])
}

// TestReplicatePostSubmit_InvalidOutputURL pins the structural credential
// guard: an output URL that is not an absolute http(s) URL is a
// configuration error, and no request is made for it.
func TestReplicatePostSubmit_InvalidOutputURL(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("no request should be made for an invalid output URL")
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", "cdn/output.png")),
		http.Header{},
	)

	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestReplicatePostSubmit_MissingOutputURL exercises the exchange when a
// succeeded prediction's output is an empty string: the output extraction
// fails and surfaces as a serialization error before any download.
func TestReplicatePostSubmit_MissingOutputURL(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("no request should be made for a missing output URL")
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", "")),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	require.ErrorContains(t, err, "output URL is empty")
}

// TestReplicatePostSubmit_PollUntilComplete exercises the full exchange when
// the prediction is still running on submit: the prediction URL is polled
// (with the caller's bearer token) until it succeeds, then the output is
// downloaded unauthenticated from the provider CDN.
func TestReplicatePostSubmit_PollUntilComplete(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

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
		case pollPath:
			gotPollAuth = r.Header.Get("Authorization")
			gotPollAccept = r.Header.Get("Accept")
			gotPollUA = r.Header.Get("User-Agent")

			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write(
					[]byte(replicatePredictionBody("processing", pollPath, "", nil)),
				)

				return
			}
			_, _ = w.Write(
				[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
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

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
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

// TestReplicatePostSubmit_PollQueryString exercises the poll-target
// construction when the provider attaches a query string to the prediction
// URL: the query rides on every poll request.
func TestReplicatePostSubmit_PollQueryString(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var (
		polls     atomic.Int64
		gotQuery  string
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/cdn/output.png" {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})

			return
		}
		if r.URL.Path != pollPath {
			t.Errorf("unexpected request path %q", r.URL.Path)

			return
		}
		gotQuery = r.URL.RawQuery

		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			_, _ = w.Write(
				[]byte(replicatePredictionBody("processing", pollPath, "", nil)),
			)

			return
		}
		_, _ = w.Write(
			[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
		)
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		// The poll target is resolved once from the submit response, so the
		// query string rides on the submit's urls.get.
		[]byte(replicatePredictionBody("starting", pollPath+"?trace=abc", "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, int64(2), polls.Load())
	// The query string the provider attached to urls.get rides on the poll.
	require.Equal(t, "trace=abc", gotQuery)
}

// TestReplicatePostSubmit_UnknownStatusKeepsPolling exercises the unknown
// terminal-status classification: a status Replicate introduces after this
// SDK shipped must not abort a running prediction, so an unrecognized status
// keeps polling on the regular schedule until a known terminal status
// arrives.
func TestReplicatePostSubmit_UnknownStatusKeepsPolling(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var (
		polls     atomic.Int64
		downloads atomic.Int64
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pollPath {
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})

			return
		}
		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			_, _ = w.Write(
				[]byte(replicatePredictionBody("queued_futures", pollPath, "", nil)),
			)

			return
		}
		_, _ = w.Write(
			[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
		)
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load(), "an unknown status keeps polling")
	require.Equal(t, int64(1), downloads.Load())
}

// TestReplicatePostSubmit_TransientPollErrors exercises the transient-error
// path: a 429 poll response (without Retry-After) falls back to the regular
// schedule and the exchange completes.
func TestReplicatePostSubmit_TransientPollErrors(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pollPath {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})

			return
		}
		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)

			return
		}
		_, _ = w.Write(
			[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
		)
	}))
	defer server.Close()
	outputURL = server.URL + "/output.png"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load())
}

// TestReplicatePostSubmit_FailedStatus exercises the terminal-failure path: a
// failed prediction stops polling and surfaces the provider's error message.
//
//nolint:dupl // Mirrors the parallel queue-provider suites; sharing a helper would couple the providers' fixtures.
func TestReplicatePostSubmit_FailedStatus(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		polls.Add(1)
		_, _ = w.Write([]byte(
			replicatePredictionBody("failed", pollPath, "CUDA out of memory", nil),
		))
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
	require.ErrorContains(t, err, "pred_123")
	require.Equal(t, int64(1), polls.Load(), "a failed prediction stops polling")
}

// TestReplicatePostSubmit_FailedOnSubmit exercises the submit short-circuit:
// a submit response that already carries a terminal failure status fails
// immediately without issuing any poll request.
func TestReplicatePostSubmit_FailedOnSubmit(t *testing.T) {
	t.Parallel()

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("failed", "", "CUDA out of memory", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
	require.ErrorContains(t, err, "pred_123")
	require.Equal(t, int64(0), polls.Load(), "a failed submit should not poll")
}

// TestReplicatePostSubmit_CanceledOnSubmit pins the second terminal failure
// status: a canceled prediction stops the exchange without issuing any poll
// request, and without a provider error message.
func TestReplicatePostSubmit_CanceledOnSubmit(t *testing.T) {
	t.Parallel()

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("canceled", "", "", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "Replicate prediction failed")
	require.ErrorContains(t, err, "pred_123")
	require.Equal(t, int64(0), polls.Load(), "a canceled prediction never polls")
}

// TestReplicatePostSubmit_ClientErrorPoll exercises the non-transient poll
// failure path: a 4xx poll response stops polling and surfaces the API error.
func TestReplicatePostSubmit_ClientErrorPoll(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	testutils.AssertAPIErrorStatus(t, err, http.StatusUnauthorized)
	require.Equal(t, int64(1), polls.Load(), "a client-error poll stops polling")
}

// TestReplicatePostSubmit_RetryAfterHonored exercises the server-provided
// wait override: a 429 poll carrying Retry-After replaces the regular
// schedule for the following round.
func TestReplicatePostSubmit_RetryAfterHonored(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var (
		polls     atomic.Int64
		downloads atomic.Int64
		outputURL string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case pollPath:
			w.Header().Set("Content-Type", "application/json")
			switch polls.Add(1) {
			case 1:
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)

				return
			case 2:
				_, _ = w.Write(
					[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
				)

				return
			}
			t.Error("polling continued past the completed status")
		default:
			downloads.Add(1)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
		}
	}))
	defer server.Close()
	outputURL = server.URL + "/cdn/output.png"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())
}

// TestReplicatePostSubmit_MalformedResponses exercises the malformed-response
// paths: non-JSON prediction bodies, completed predictions without outputs,
// missing prediction URLs, and — on the poll side — non-JSON and oversized
// bodies are errors that stop polling.
func TestReplicatePostSubmit_MalformedResponses(t *testing.T) {
	t.Parallel()

	t.Run("non-JSON submit body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
			}),
		)
		defer server.Close()

		codec := fastReplicateImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte("not json"),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("succeeded without output", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := fastReplicateImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(replicatePredictionBody("succeeded", "", "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without output")
	})

	t.Run("missing prediction URL", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := fastReplicateImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(replicatePredictionBody("starting", "", "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "no prediction URL")
	})

	t.Run("non-JSON poll body", func(t *testing.T) {
		t.Parallel()

		const pollPath = "/v1/predictions/pred_123"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != pollPath {
				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("not json"))
		}))
		defer server.Close()

		codec := fastReplicateImageCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode Replicate prediction response")
		require.Equal(t, int64(1), polls.Load(), "a malformed poll stops polling")
	})

	t.Run("oversized poll body", func(t *testing.T) {
		t.Parallel()

		const pollPath = "/v1/predictions/pred_123"

		var polls atomic.Int64
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != pollPath {
				return
			}
			polls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(
				[]byte(replicatePredictionBody("processing", pollPath, "", nil)),
			)
		}))
		defer server.Close()

		codec := fastReplicateImageCodec()
		params := postSubmitParams(server)
		params.MaxResponseBodyBytes = 4
		_, err := codec.PostSubmit(
			context.Background(),
			params,
			[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "exceeds max size")
		require.Equal(t, int64(1), polls.Load(), "an oversized poll stops polling")
	})
}

// TestReplicatePostSubmit_MediaTypeMismatch exercises the output validation:
// each media-output codec rejects a downloaded output whose media type
// belongs to another family.
func TestReplicatePostSubmit_MediaTypeMismatch(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		postSubmit  func(server *httptest.Server) error
		outputType  string
		outputBytes []byte
		wantPrefix  string
	}{
		{
			name: "image codec gets video",
			postSubmit: func(server *httptest.Server) error {
				_, err := fastReplicateImageCodec().PostSubmit(
					context.Background(),
					postSubmitParams(server),
					[]byte(replicatePredictionBody("succeeded", "", "", server.URL+"/output")),
					http.Header{},
				)

				return err
			},
			outputType:  "video/mp4",
			outputBytes: []byte("not an image"),
			wantPrefix:  "image/",
		},
		{
			name: "video codec gets image",
			postSubmit: func(server *httptest.Server) error {
				_, err := fastReplicateVideoCodec().PostSubmit(
					context.Background(),
					postSubmitParams(server),
					[]byte(replicatePredictionBody("succeeded", "", "", server.URL+"/output")),
					http.Header{},
				)

				return err
			},
			outputType:  "image/png",
			outputBytes: []byte{0x89, 'P', 'N', 'G'},
			wantPrefix:  "video/",
		},
		{
			name: "image-to-image codec gets video",
			postSubmit: func(server *httptest.Server) error {
				_, err := fastReplicateImageToImageCodec().PostSubmit(
					context.Background(),
					postSubmitParams(server),
					[]byte(replicatePredictionBody("succeeded", "", "", server.URL+"/output")),
					http.Header{},
				)

				return err
			},
			outputType:  "video/mp4",
			outputBytes: []byte("not an image"),
			wantPrefix:  "image/",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// The /output path serves the wrong-family blob; everything
			// else serves a succeeded prediction pointing at it.
			var predURL string
			handler := func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/output" {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(
						replicatePredictionBody("succeeded", "", "", predURL),
					))

					return
				}
				w.Header().Set("Content-Type", tc.outputType)
				_, _ = w.Write(tc.outputBytes)
			}
			server := httptest.NewServer(http.HandlerFunc(handler))
			defer server.Close()
			predURL = server.URL + "/output"

			err := tc.postSubmit(server)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
			require.ErrorContains(t, err, tc.wantPrefix)
		})
	}
}

// TestReplicatePostSubmit_ImageToImage_Complete mirrors the text-to-image
// complete-submit exchange for the image-to-image codec: a succeeded
// prediction downloads its output and returns the image bytes.
func TestReplicatePostSubmit_ImageToImage_Complete(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				replicatePredictionBody("succeeded", "", "", outputURL),
			))

			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastReplicateImageToImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", outputURL)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, "image/png", resp.MediaType)
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
}

// TestReplicatePostSubmit_ImageToImage_FailedPrediction mirrors the ASR
// failure path for the image-to-image codec: a failed prediction surfaces
// the provider's error.
func TestReplicatePostSubmit_ImageToImage_FailedPrediction(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(
			replicatePredictionBody("failed", "", "CUDA out of memory", nil),
		))
	}))
	defer server.Close()

	codec := fastReplicateImageToImageCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("failed", "", "CUDA out of memory", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
}

// TestReplicatePostSubmit_TextToVideo_Complete mirrors the text-to-image
// complete-submit exchange for the text-to-video codec: a succeeded
// prediction downloads its output and returns the video bytes.
func TestReplicatePostSubmit_TextToVideo_Complete(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				replicatePredictionBody("succeeded", "", "", outputURL),
			))

			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("not really an mp4"))
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastReplicateVideoCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", outputURL)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte("not really an mp4"), resp.Video)
	require.Equal(t, "video/mp4", resp.MediaType)
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
}

// TestReplicatePostSubmit_TextToVideo_FailedPrediction mirrors the ASR
// failure path for the text-to-video codec: a failed prediction surfaces the
// provider's error.
func TestReplicatePostSubmit_TextToVideo_FailedPrediction(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(
			replicatePredictionBody("failed", "", "CUDA out of memory", nil),
		))
	}))
	defer server.Close()

	codec := fastReplicateVideoCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("failed", "", "CUDA out of memory", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "CUDA out of memory")
}

// TestReplicatePostSubmit_SniffedMediaType exercises the media-type fallback:
// when the CDN omits Content-Type, the bytes are sniffed.
func TestReplicatePostSubmit_SniffedMediaType(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				replicatePredictionBody("succeeded", "", "", outputURL),
			))

			return
		}
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", outputURL)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
}

// TestReplicatePostSubmit_GarbledContentTypeSniffed exercises the download's
// Content-Type fallback: a garbled Content-Type header is replaced by the
// sniffed media type, which the codec then accepts.
func TestReplicatePostSubmit_GarbledContentTypeSniffed(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/output" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(
				replicatePredictionBody("succeeded", "", "", outputURL),
			))

			return
		}
		w.Header().Set("Content-Type", "%%%")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", outputURL)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "image/png", resp.MediaType)
}

// TestReplicatePostSubmit_ContextCanceled exercises cancellation: a canceled
// context aborts the exchange with the context error.
func TestReplicatePostSubmit_ContextCanceled(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	codec := fastReplicateImageCodec()
	_, err := codec.PostSubmit(
		ctx,
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	require.ErrorIs(t, err, context.Canceled)
}

// TestReplicatePostSubmit_ContextCanceledDuringPoll exercises cancellation
// after polling has started: the context is canceled while the second poll
// request is in flight, and the resulting transport error — which wraps
// context.Canceled — must surface immediately instead of being swallowed as
// a transient failure worth retrying.
func TestReplicatePostSubmit_ContextCanceledDuringPoll(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// pollStarted is closed when the second poll arrives; release unblocks
	// its handler so the test can cancel the context mid-roundtrip.
	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pollPath {
			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(replicatePredictionBody("processing", pollPath, "", nil)))
	}))
	defer server.Close()

	type posted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan posted, 1)
	codec := fastReplicateImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			postSubmitParams(server),
			[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
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

// TestReplicatePostSubmit_DeadlineExceededDuringPoll exercises deadline expiry
// after polling has started: the caller's deadline lapses while a poll request
// is in flight, and the resulting transport error — which wraps
// context.DeadlineExceeded — must surface immediately instead of being
// swallowed as a transient failure worth retrying.
func TestReplicatePostSubmit_DeadlineExceededDuringPoll(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// pollStarted is closed when the second poll arrives; release unblocks
	// its handler after the deadline has lapsed, so the in-flight poll dies
	// with a context error.
	pollStarted := make(chan struct{})
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pollPath {
			return
		}

		if polls.Add(1) == 2 {
			close(pollStarted)
			<-release
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(replicatePredictionBody("processing", pollPath, "", nil)))
	}))
	defer server.Close()

	type posted struct {
		resp hftypes.TextToImageResponse
		err  error
	}
	results := make(chan posted, 1)
	codec := fastReplicateImageCodec()
	go func() {
		resp, err := codec.PostSubmit(
			ctx,
			postSubmitParams(server),
			[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
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

// TestReplicatePostSubmit_ForeignPredictionURLCannotRedirect pins the poll
// invariant end to end: a provider-supplied absolute prediction URL pointing
// at a foreign host cannot redirect the token-bearing poll — the host is
// discarded and the path resolves against the caller's base URL.
func TestReplicatePostSubmit_ForeignPredictionURLCannotRedirect(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	foreign := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("a provider-supplied absolute prediction URL must not redirect the poll")
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
		case pollPath:
			gotPollAuth = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			if polls.Add(1) == 1 {
				_, _ = w.Write(
					[]byte(replicatePredictionBody("processing", pollPath, "", nil)),
				)

				return
			}
			_, _ = w.Write(
				[]byte(replicatePredictionBody("succeeded", pollPath, "", outputURL)),
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

	codec := fastReplicateImageCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		// The submit response carries an absolute prediction URL pointing
		// at a foreign host; the poll must still resolve to the caller's
		// host.
		[]byte(replicatePredictionBody("starting", foreign.URL+pollPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, []byte{0x89, 'P', 'N', 'G'}, resp.Image)
	require.Equal(t, int64(2), polls.Load())
	require.Equal(t, int64(1), downloads.Load())
	// The token stays on the caller's base host, never the foreign host.
	require.Equal(t, "Bearer test-token", gotPollAuth)
}

// TestReplicatePostSubmit_DownloadBodyLimit exercises the download-side
// response-size cap: a succeeded prediction whose CDN output exceeds
// PostSubmitParams.MaxResponseBodyBytes fails instead of downloading without
// bound.
func TestReplicatePostSubmit_DownloadBodyLimit(t *testing.T) {
	t.Parallel()

	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/output" {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', 0, 0, 0, 0, 0, 0, 0, 0})

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(
			replicatePredictionBody("succeeded", "", "", outputURL),
		))
	}))
	defer server.Close()
	outputURL = server.URL + "/output"

	codec := fastReplicateImageCodec()
	params := postSubmitParams(server)
	params.MaxResponseBodyBytes = 4
	_, err := codec.PostSubmit(
		context.Background(),
		params,
		[]byte(replicatePredictionBody("succeeded", "", "", outputURL)),
		http.Header{},
	)
	require.Error(t, err)
	require.ErrorContains(t, err, "exceeds max size")
}

// TestReplicateSpeechRecognitionPostSubmit_PollUntilTranscript exercises the
// ASR exchange when the prediction is still running on submit: the
// prediction URL is polled until it succeeds, and the transcript — carried
// directly in the prediction output — surfaces without a download.
func TestReplicateSpeechRecognitionPostSubmit_PollUntilTranscript(t *testing.T) {
	t.Parallel()

	const pollPath = "/v1/predictions/pred_123"

	var polls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pollPath {
			t.Errorf("unexpected request path %q", r.URL.Path)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		if polls.Add(1) == 1 {
			_, _ = w.Write(
				[]byte(replicatePredictionBody("processing", pollPath, "", nil)),
			)

			return
		}
		_, _ = w.Write(
			[]byte(replicatePredictionBody("succeeded", pollPath, "", "hello world")),
		)
	}))
	defer server.Close()

	codec := fastReplicateSpeechCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("starting", pollPath, "", nil)),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "hello world", resp.Text)
	require.Nil(t, resp.Chunks, "Replicate output carries no chunk timestamps")
	require.Equal(t, "pred_123", resp.ProviderMetadata["id"])
	require.Equal(t, "succeeded", resp.ProviderMetadata["status"])
	require.Equal(t, int64(2), polls.Load())
}

// TestReplicateSpeechRecognitionPostSubmit_TranscriptShapes pins every
// transcript output shape the official clients accept, all completing on
// submit: a plain string, a one-element list, and the multilingual object
// carrying transcription, translation, or both (transcription wins).
func TestReplicateSpeechRecognitionPostSubmit_TranscriptShapes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		output any
		want   string
	}{
		{
			name:   "string output",
			output: "hello world",
			want:   "hello world",
		},
		{
			name:   "one-element list output",
			output: []string{"hello world"},
			want:   "hello world",
		},
		{
			name:   "transcription object",
			output: map[string]any{"transcription": "bonjour"},
			want:   "bonjour",
		},
		{
			name:   "translation object",
			output: map[string]any{"translation": "hello"},
			want:   "hello",
		},
		{
			name: "transcription wins over translation",
			output: map[string]any{
				"transcription": "bonjour",
				"translation":   "hello",
			},
			want: "bonjour",
		},
	}
	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					t.Error("no request should be made")
				}),
			)
			defer server.Close()

			codec := fastReplicateSpeechCodec()
			resp, err := codec.PostSubmit(
				context.Background(),
				postSubmitParams(server),
				[]byte(replicatePredictionBody("succeeded", "", "", tc.output)),
				http.Header{},
			)
			require.NoError(t, err)
			require.Equal(t, tc.want, resp.Text)
		})
	}
}

// TestReplicateSpeechRecognitionPostSubmit_TxtFileDownload exercises the
// txt_file output shape: the transcript lives on a provider-hosted text file
// that is downloaded unauthenticated.
func TestReplicateSpeechRecognitionPostSubmit_TxtFileDownload(t *testing.T) {
	t.Parallel()

	var (
		downloads     atomic.Int64
		gotDLAuth     string
		gotDLUA       string
		gotDLAccept   string
		gotDLResponse string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/transcript.txt" {
			t.Errorf("unexpected request path %q", r.URL.Path)

			return
		}
		downloads.Add(1)
		gotDLAuth = r.Header.Get("Authorization")
		gotDLUA = r.Header.Get("User-Agent")
		gotDLAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte("the quick brown fox"))
	}))
	defer server.Close()

	codec := fastReplicateSpeechCodec()
	resp, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("succeeded", "", "", map[string]any{
			"txt_file": server.URL + "/transcript.txt",
		})),
		http.Header{},
	)
	require.NoError(t, err)
	require.Equal(t, "the quick brown fox", resp.Text)
	require.Equal(t, int64(1), downloads.Load())
	// The download is unauthenticated and reuses the caller's User-Agent.
	require.Empty(t, gotDLAuth)
	require.Equal(t, "test-agent/1.0", gotDLUA)
	require.Equal(t, "*/*", gotDLAccept)
	require.Empty(t, gotDLResponse)
}

// TestReplicateSpeechRecognitionPostSubmit_TxtFileDownloadErrors exercises
// the txt_file download's failure paths: a server error on the txt URL and
// an oversized body both abort the exchange.
func TestReplicateSpeechRecognitionPostSubmit_TxtFileDownloadErrors(t *testing.T) {
	t.Parallel()

	t.Run("server error on the txt URL", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/transcript.txt" {
				t.Errorf("unexpected request path %q", r.URL.Path)

				return
			}
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer server.Close()

		codec := fastReplicateSpeechCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(replicatePredictionBody("succeeded", "", "", map[string]any{
				"txt_file": server.URL + "/transcript.txt",
			})),
			http.Header{},
		)
		testutils.AssertAPIErrorStatus(t, err, http.StatusInternalServerError)
	})

	t.Run("oversized txt body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/transcript.txt" {
				t.Errorf("unexpected request path %q", r.URL.Path)

				return
			}
			_, _ = w.Write([]byte("the quick brown fox"))
		}))
		defer server.Close()

		params := postSubmitParams(server)
		params.MaxResponseBodyBytes = 4

		codec := fastReplicateSpeechCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			params,
			[]byte(replicatePredictionBody("succeeded", "", "", map[string]any{
				"txt_file": server.URL + "/transcript.txt",
			})),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "exceeds max size")
	})
}

// TestReplicateSpeechRecognitionPostSubmit_MalformedOutput exercises the
// malformed-output paths: a non-transcript object, an empty output list, and
// a numeric output are serialization errors; a succeeded prediction without
// output is rejected by the terminal-status classification.
func TestReplicateSpeechRecognitionPostSubmit_MalformedOutput(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		output any
	}{
		{
			name:   "object without transcript fields",
			output: map[string]any{"foo": "bar"},
		},
		{
			name:   "empty output list",
			output: []string{},
		},
		{
			name:   "numeric output",
			output: 42,
		},
	}
	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					t.Error("no request should be made")
				}),
			)
			defer server.Close()

			codec := fastReplicateSpeechCodec()
			_, err := codec.PostSubmit(
				context.Background(),
				postSubmitParams(server),
				[]byte(replicatePredictionBody("succeeded", "", "", tc.output)),
				http.Header{},
			)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
			require.ErrorContains(
				t,
				err,
				"malformed response from Replicate automatic-speech-recognition API",
			)
		})
	}

	t.Run("succeeded without transcript", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(
			http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
		)
		defer server.Close()

		codec := fastReplicateSpeechCodec()
		_, err := codec.PostSubmit(
			context.Background(),
			postSubmitParams(server),
			[]byte(replicatePredictionBody("succeeded", "", "", nil)),
			http.Header{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without output")
	})
}

// TestReplicateSpeechRecognitionPostSubmit_FailedPrediction exercises the
// ASR failure path: a failed prediction surfaces the provider's error.
func TestReplicateSpeechRecognitionPostSubmit_FailedPrediction(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}),
	)
	defer server.Close()

	codec := fastReplicateSpeechCodec()
	_, err := codec.PostSubmit(
		context.Background(),
		postSubmitParams(server),
		[]byte(replicatePredictionBody("failed", "", "audio decoding failed", nil)),
		http.Header{},
	)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "audio decoding failed")
}

// TestReplicatePredictionRoute exercises the endpoint routing: plain
// identifiers address the model-scoped route; version-pinned identifiers
// submit through the versionless predictions route.
func TestReplicatePredictionRoute(t *testing.T) {
	t.Parallel()

	t.Run("model-scoped route", func(t *testing.T) {
		t.Parallel()

		require.Equal(
			t,
			"replicate/v1/models/black-forest-labs/flux-dev/predictions",
			replicatePredictionRoute("black-forest-labs/flux-dev"),
		)
	})

	t.Run("version-pinned route", func(t *testing.T) {
		t.Parallel()

		require.Equal(
			t,
			"replicate/v1/predictions",
			replicatePredictionRoute("black-forest-labs/flux-dev:abc123def"),
		)
	})

	t.Run("trailing colon stays model-scoped", func(t *testing.T) {
		t.Parallel()

		require.Equal(
			t,
			"replicate/v1/models/black-forest-labs/flux-dev:/predictions",
			replicatePredictionRoute("black-forest-labs/flux-dev:"),
		)
	})
}

// TestReplicatePollTarget exercises the poll-target resolution: the Hugging
// Face router gains a /replicate prefix, already-routed URLs keep their
// path, other hosts resolve as-is, queries ride along, and missing or
// unparseable inputs fail with their respective kinds.
func TestReplicatePollTarget(t *testing.T) {
	t.Parallel()

	pollTarget := func(t *testing.T, baseURL, getURL string) (request.Params, string) {
		t.Helper()

		pollParams, pollPath, err := replicatePollTarget(
			PostSubmitParams{BaseURL: baseURL},
			&replicatePrediction{URLs: replicatePredictionURLs{Get: getURL}},
		)
		require.NoError(t, err)

		return pollParams, pollPath
	}

	t.Run("router host gains /replicate prefix", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://router.huggingface.co", "/v1/predictions/pred_123",
		)
		require.Equal(t, "https://router.huggingface.co/replicate", pollParams.BaseURL)
		require.Equal(t, "/v1/predictions/pred_123", pollPath)
	})

	t.Run("router host matched case-insensitively", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://Router.HuggingFace.CO", "/v1/predictions/pred_123",
		)
		require.Equal(t, "https://Router.HuggingFace.CO/replicate", pollParams.BaseURL)
		require.Equal(t, "/v1/predictions/pred_123", pollPath)
	})

	t.Run("already-routed URL keeps its path", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t,
			"https://router.huggingface.co",
			"https://router.huggingface.co/replicate/v1/predictions/pred_123",
		)
		require.Equal(t, "https://router.huggingface.co", pollParams.BaseURL)
		require.Equal(t, "/replicate/v1/predictions/pred_123", pollPath)
	})

	t.Run("other hosts resolve as-is", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t, "https://api.example.com", "/v1/predictions/pred_123",
		)
		require.Equal(t, "https://api.example.com", pollParams.BaseURL)
		require.Equal(t, "/v1/predictions/pred_123", pollPath)
	})

	t.Run("trailing slash trimmed before prefixing", func(t *testing.T) {
		t.Parallel()

		pollParams, _ := pollTarget(
			t, "https://router.huggingface.co/", "/v1/predictions/pred_123",
		)
		require.Equal(t, "https://router.huggingface.co/replicate", pollParams.BaseURL)
	})

	t.Run("query string appended to the path", func(t *testing.T) {
		t.Parallel()

		_, pollPath := pollTarget(
			t, "https://router.huggingface.co", "/v1/predictions/pred_123?trace=1",
		)
		require.Equal(t, "/v1/predictions/pred_123?trace=1", pollPath)
	})

	t.Run("absolute get URL resolved from its path", func(t *testing.T) {
		t.Parallel()

		pollParams, pollPath := pollTarget(
			t,
			"https://router.huggingface.co",
			"https://api.replicate.com/v1/predictions/pred_123",
		)
		require.Equal(t, "https://router.huggingface.co/replicate", pollParams.BaseURL)
		require.Equal(t, "/v1/predictions/pred_123", pollPath)
	})

	t.Run("relative get path gains a leading slash", func(t *testing.T) {
		t.Parallel()

		_, pollPath := pollTarget(
			t, "https://router.huggingface.co", "v1/predictions/pred_123",
		)
		require.Equal(t, "/v1/predictions/pred_123", pollPath)
	})

	t.Run("empty prediction URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := replicatePollTarget(
			PostSubmitParams{BaseURL: "https://router.huggingface.co"},
			&replicatePrediction{},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("invalid base URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := replicatePollTarget(
			PostSubmitParams{BaseURL: "http://[::1:bad"},
			&replicatePrediction{URLs: replicatePredictionURLs{Get: "/v1/predictions/pred_123"}},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("invalid prediction URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := replicatePollTarget(
			PostSubmitParams{BaseURL: "https://router.huggingface.co"},
			&replicatePrediction{URLs: replicatePredictionURLs{Get: "http://h/%zz"}},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestReplicateMetadata exercises provider-metadata extraction: an empty
// prediction yields nil so callers never observe an empty map.
func TestReplicateMetadata(t *testing.T) {
	t.Parallel()

	t.Run("empty prediction", func(t *testing.T) {
		t.Parallel()

		require.Nil(t, replicateMetadata(&replicatePrediction{}))
	})

	t.Run("populated prediction", func(t *testing.T) {
		t.Parallel()

		meta := replicateMetadata(&replicatePrediction{
			ID:        "pred_1",
			Model:     "black-forest-labs/flux-dev",
			Status:    "succeeded",
			CreatedAt: "2026-01-01T00:00:00Z",
			Logs:      "running on H100",
			Metrics:   &replicateMetrics{PredictTime: new(1.5)},
		})
		require.Equal(t, "pred_1", meta["id"])
		require.Equal(t, "black-forest-labs/flux-dev", meta["model"])
		require.Equal(t, "succeeded", meta["status"])
		require.Equal(t, "2026-01-01T00:00:00Z", meta["created_at"])
		require.Equal(t, "running on H100", meta["logs"])
		require.Equal(t, map[string]any{"predict_time": 1.5}, meta["metrics"])
	})

	t.Run("logs omitted when empty", func(t *testing.T) {
		t.Parallel()

		meta := replicateMetadata(&replicatePrediction{ID: "pred_2", Status: "succeeded"})
		require.Equal(t, "pred_2", meta["id"])
		require.NotContains(t, meta, "logs")
		require.NotContains(t, meta, "model")
	})

	t.Run("timing fields omitted when absent", func(t *testing.T) {
		t.Parallel()

		meta := replicateMetadata(&replicatePrediction{ID: "pred_3", Status: "succeeded"})
		require.NotContains(t, meta, "created_at")
		require.NotContains(t, meta, "metrics")
	})

	t.Run("empty metrics breakdown omitted", func(t *testing.T) {
		t.Parallel()

		meta := replicateMetadata(&replicatePrediction{
			ID:      "pred_4",
			Metrics: &replicateMetrics{},
		})
		require.Equal(t, "pred_4", meta["id"])
		require.NotContains(t, meta, "metrics")
	})
}

// TestReplicatePredictionFailedError exercises the terminal-failure error
// message: the provider's error message and prediction ID are appended only
// when present.
func TestReplicatePredictionFailedError(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		pred replicatePrediction
		want string
	}{
		{
			name: "bare failure",
			pred: replicatePrediction{Status: replicateStatusFailed},
			want: "Replicate prediction failed",
		},
		{
			name: "with provider error",
			pred: replicatePrediction{
				Status: replicateStatusFailed,
				Error:  new("CUDA out of memory"),
			},
			want: "Replicate prediction failed: CUDA out of memory",
		},
		{
			name: "with prediction id",
			pred: replicatePrediction{Status: replicateStatusFailed, ID: "pred_1"},
			want: "Replicate prediction failed (prediction id pred_1)",
		},
		{
			name: "with provider error and prediction id",
			pred: replicatePrediction{
				Status: replicateStatusFailed,
				ID:     "pred_1",
				Error:  new("CUDA out of memory"),
			},
			want: "Replicate prediction failed: CUDA out of memory (prediction id pred_1)",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := replicatePredictionFailedError(&tc.pred)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

// TestReplicateMediaOutputURL exercises the completed-prediction output
// extraction: string and list outputs return their first URL, while missing,
// empty, and non-URL outputs are serialization errors.
func TestReplicateMediaOutputURL(t *testing.T) {
	t.Parallel()

	t.Run("string output", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{
			Status: replicateStatusSucceeded,
			Output: json.RawMessage(`"https://cdn.example.com/output.png"`),
		}

		url, err := replicateMediaOutputURL(&pred)
		require.NoError(t, err)
		require.Equal(t, "https://cdn.example.com/output.png", url)
	})

	t.Run("list output takes the first URL", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{
			Status: replicateStatusSucceeded,
			Output: json.RawMessage(
				`["https://cdn.example.com/a.png","https://cdn.example.com/b.png"]`,
			),
		}

		url, err := replicateMediaOutputURL(&pred)
		require.NoError(t, err)
		require.Equal(t, "https://cdn.example.com/a.png", url)
	})

	t.Run("without output", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{Status: replicateStatusSucceeded}
		_, err := replicateMediaOutputURL(&pred)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "without output")
	})

	t.Run("empty string output", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{
			Status: replicateStatusSucceeded,
			Output: json.RawMessage(`""`),
		}
		_, err := replicateMediaOutputURL(&pred)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "URL is empty")
	})

	t.Run("empty list output", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{
			Status: replicateStatusSucceeded,
			Output: json.RawMessage(`[]`),
		}
		_, err := replicateMediaOutputURL(&pred)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "has no URL")
	})

	t.Run("object output", func(t *testing.T) {
		t.Parallel()

		pred := replicatePrediction{
			Status: replicateStatusSucceeded,
			Output: json.RawMessage(`{"foo":"bar"}`),
		}
		_, err := replicateMediaOutputURL(&pred)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "neither a URL nor a list of URLs")
	})
}

// TestReplicateInputMIME exercises the input media-type labeling: sniffed
// image and audio containers keep their sniffed type, and unrecognized
// bytes fall back to the official clients' defaults.
func TestReplicateInputMIME(t *testing.T) {
	t.Parallel()

	t.Run("audio", func(t *testing.T) {
		t.Parallel()

		require.Equal(t, "audio/wave", replicateAudioMIME([]byte(
			"RIFF\x00\x00\x00\x00WAVE",
		)))
		require.Equal(t, "audio/flac", replicateAudioMIME([]byte("fLaC")))
		// Unrecognized bytes fall back to the official clients' default.
		require.Equal(t, "audio/wav", replicateAudioMIME([]byte("abc123")))
	})

	t.Run("image", func(t *testing.T) {
		t.Parallel()

		require.Equal(
			t,
			"image/png",
			replicateImageMIME([]byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}),
		)
		require.Equal(
			t,
			"image/jpeg",
			replicateImageMIME([]byte{0xFF, 0xD8, 0xFF, 0xE0}),
		)
		// Unrecognized bytes fall back to the official clients' default.
		require.Equal(t, "image/jpeg", replicateImageMIME([]byte("abc123")))
	})
}

// TestReplicateClassifyPrediction exercises the status classification: a
// succeeded prediction with output completes, succeeded without output and
// the terminal failure statuses complete with errors, and every other status
// — including unknown ones — keeps polling.
func TestReplicateClassifyPrediction(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		status   string
		output   json.RawMessage
		done     bool
		wantKind hferrors.SDKErrorKind
		wantErr  bool
	}{
		{
			name:   "succeeded with output",
			status: replicateStatusSucceeded,
			output: json.RawMessage(`"https://cdn.example.com/a.png"`),
			done:   true,
		},
		{
			name:     "succeeded without output",
			status:   replicateStatusSucceeded,
			output:   json.RawMessage("null"),
			done:     true,
			wantErr:  true,
			wantKind: hferrors.SDKErrorKindSerialization,
		},
		{
			name:     "failed",
			status:   replicateStatusFailed,
			done:     true,
			wantErr:  true,
			wantKind: hferrors.SDKErrorKindProvider,
		},
		{
			name:     "canceled",
			status:   replicateStatusCanceled,
			done:     true,
			wantErr:  true,
			wantKind: hferrors.SDKErrorKindProvider,
		},
		{name: "starting keeps polling", status: "starting"},
		{name: "processing keeps polling", status: "processing"},
		{name: "unknown status keeps polling", status: "in_queue"},
		{name: "empty status keeps polling"},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			pred := replicatePrediction{Status: tc.status, Output: tc.output}
			out, done, err := replicateTerminalPrediction(&pred)
			require.Equal(t, tc.done, done)

			if !tc.wantErr {
				require.NoError(t, err)

				if tc.done {
					// A completed prediction returns the prediction itself
					// so callers can extract its output.
					require.Same(t, &pred, out)
				} else {
					require.Nil(t, out)
				}

				return
			}

			testutils.AssertSDKErrorKind(t, err, tc.wantKind)
			require.Nil(t, out)
		})
	}
}

// TestReplicateEncodeHeaders exercises the shared submission headers: a JSON
// body whose submit response is always JSON, with the "Prefer: wait"
// directive.
func TestReplicateEncodeHeaders(t *testing.T) {
	t.Parallel()

	headers := replicateEncodeHeaders()
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "application/json", headers.Get("Accept"))
	require.Equal(t, "wait", headers.Get("Prefer"))
}

// TestReplicateCodec_EncodeUnmarshalableProviderParams exercises every task
// codec's encode-side marshal guard: a ProviderParams entry json cannot
// encode makes the whole prediction payload unmarshalable, which surfaces as
// a serialization error instead of a garbled request.
func TestReplicateCodec_EncodeUnmarshalableProviderParams(t *testing.T) {
	t.Parallel()

	t.Run("text-to-video", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToVideoCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
			Context: context.Background(),
			Request: hftypes.TextToVideoRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	})

	t.Run("text-to-image", func(t *testing.T) {
		t.Parallel()

		codec := replicateTextToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "black-forest-labs/FLUX.1-dev",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	})

	t.Run("image-to-image", func(t *testing.T) {
		t.Parallel()

		codec := replicateImageToImageCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
			Context: context.Background(),
			Request: hftypes.ImageToImageRequest{
				Input:          "data:image/png;base64,abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "bria/fibo-2.0/edit",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	})

	t.Run("speech-recognition", func(t *testing.T) {
		t.Parallel()

		codec := replicateSpeechRecognitionCodec{}
		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          "data:audio/wav;base64,abc123",
				ProviderParams: map[string]any{"bad": func() {}},
			},
			Model: "openai/whisper-large-v3",
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	})
}
