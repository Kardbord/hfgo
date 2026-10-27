//go:build !integration

package hfproviders

import (
	"context"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

func TestWavespeedProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	require.Equal(t, "wavespeed", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

// requireNilResolverError asserts err is the configuration error returned for
// a missing WaveSpeed model resolver.
func requireNilResolverError(t *testing.T, err error) {
	t.Helper()

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

func TestNewWavespeedProvider_NilResolver(t *testing.T) {
	t.Parallel()

	p, err := NewWavespeedProvider(nil)
	requireNilResolverError(t, err)
	require.Equal(t, WavespeedProvider{}, p)
}

func TestWavespeedProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var p WavespeedProvider

	_, err := p.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)

	_, err = p.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)

	_, err = p.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)
}

func TestWavespeedProvider_TextToVideoEndpoint(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(map[StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     TaskTextToVideo,
			Model:    "ZhengmingYu/DMAD",
		}: "wavespeed-ai/minimax-h3/text-to-video-lora",
	})
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	endpoint, err := p.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "ZhengmingYu/DMAD",
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "wavespeed/wavespeed-ai/minimax-h3/text-to-video-lora", endpoint.Path)
}

func TestWavespeedProvider_TextToImageEndpoint(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(map[StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     TaskTextToImage,
			Model:    "black-forest-labs/FLUX.1-dev",
		}: "wavespeed-ai/flux-dev",
	})
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	endpoint, err := p.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "black-forest-labs/FLUX.1-dev",
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "wavespeed/wavespeed-ai/flux-dev", endpoint.Path)
}

func TestWavespeedProvider_ImageToImageEndpoint(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(map[StaticMapping]string{
		{
			Provider: "wavespeed",
			Task:     TaskImageToImage,
			Model:    "briaai/Fibo-2.0-edit-turbo",
		}: "bria/fibo-2.0/edit",
	})
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	endpoint, err := p.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "briaai/Fibo-2.0-edit-turbo",
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "wavespeed/bria/fibo-2.0/edit", endpoint.Path)
}

func TestWavespeedProvider_Endpoint_ResolutionError(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	_, err = p.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "unknown-model",
	})
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
}

func TestWavespeedProvider_TextToVideoCodec_Encode(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)
	codec := p.TextToVideoCodec()

	body, headers, err := codec.Encode(EncodeParams[hftypes.TextToVideoRequest]{
		Context: context.Background(),
		Request: hftypes.TextToVideoRequest{
			Input: "a cat",
			Parameters: &hftypes.TextToVideoParameters{
				NumFrames: new(24),
			},
		},
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"inputs":"a cat","parameters":{"num_frames":24}}`, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "*/*", headers.Get("Accept"))
}

func TestWavespeedProvider_TextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)
	codec := p.TextToImageCodec()

	body, headers, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
		Context: context.Background(),
		Request: hftypes.TextToImageRequest{
			Input: "a cat",
			Parameters: &hftypes.TextToImageParameters{
				Width: new(512),
			},
		},
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"inputs":"a cat","parameters":{"width":512}}`, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "*/*", headers.Get("Accept"))
}

func TestWavespeedProvider_ImageToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)
	codec := p.ImageToImageCodec()

	body, headers, err := codec.Encode(EncodeParams[hftypes.ImageToImageRequest]{
		Context: context.Background(),
		Request: hftypes.ImageToImageRequest{
			Input: "data:image/png;base64,abc123",
			Parameters: &hftypes.ImageToImageParameters{
				Prompt: new("make it blue"),
			},
		},
	})
	require.NoError(t, err)
	require.JSONEq(
		t,
		`{"inputs":"data:image/png;base64,abc123","parameters":{"prompt":"make it blue"}}`,
		string(body),
	)
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "*/*", headers.Get("Accept"))
}

func TestWavespeedProvider_TextToVideoCodec_Decode(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)
	codec := p.TextToVideoCodec()

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

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
	})
}

// assertImageMediaDecode exercises an image-output codec's Decode: a valid
// image response is returned with its media type, and a non-image Content-Type
// is rejected as a serialization error. getPayload extracts the image bytes
// and advertised media type from the response.
func assertImageMediaDecode[Res any](
	t *testing.T,
	decode func(DecodeParams) (Res, error),
	getPayload func(Res) ([]byte, string),
) {
	t.Helper()

	t.Run("valid image", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "image/png; charset=utf-8")
		body := []byte{0x89, 'P', 'N', 'G'}

		resp, err := decode(DecodeParams{
			Context: context.Background(),
			Body:    body,
			Headers: headers,
		})
		require.NoError(t, err)

		image, mediaType := getPayload(resp)
		require.Equal(t, body, image)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("invalid content type", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		_, err := decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"foo":"bar"}`),
			Headers: headers,
		})
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
	})
}

// TestWavespeedProvider_ImageCodecDecodes exercises the image-output codecs
// shared by the text-to-image and image-to-image mixins.
func TestWavespeedProvider_ImageCodecDecodes(t *testing.T) {
	t.Parallel()

	mapper := NewStaticModelResolver(nil)
	p, err := NewWavespeedProvider(mapper)
	require.NoError(t, err)

	t.Run("text-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := p.TextToImageCodec()
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.TextToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})

	t.Run("image-to-image codec", func(t *testing.T) {
		t.Parallel()

		codec := p.ImageToImageCodec()
		assertImageMediaDecode(t, codec.Decode,
			func(r hftypes.ImageToImageResponse) ([]byte, string) {
				return r.Image, r.MediaType
			})
	})
}

// wavespeedSubsetProvider serves only text-to-image by embedding the
// corresponding WaveSpeed mixin with a resolver set directly, without going
// through NewWavespeedProvider.
type wavespeedSubsetProvider struct {
	WavespeedTextToImageMixin
}

func (wavespeedSubsetProvider) ProviderSuffix() string { return "" }

func (wavespeedSubsetProvider) Name() string { return "subset" }

func TestWavespeedMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	p := wavespeedSubsetProvider{
		WavespeedTextToImageMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]string{
				{
					Provider: "wavespeed",
					Task:     TaskTextToImage,
					Model:    "black-forest-labs/FLUX.1-dev",
				}: "wavespeed-ai/flux-dev",
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
	require.Equal(t, "wavespeed/wavespeed-ai/flux-dev", endpoint.Path)

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

func TestWavespeedMixins_ZeroValueResolvers(t *testing.T) {
	t.Parallel()

	var (
		videoMixin  WavespeedTextToVideoMixin
		imageMixin  WavespeedTextToImageMixin
		image2Mixin WavespeedImageToImageMixin
	)

	_, err := videoMixin.TextToVideoEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)

	_, err = imageMixin.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)

	_, err = image2Mixin.ImageToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	requireNilResolverError(t, err)
}

// TestWavespeedTextToImageMixin_ReusesHFCodec asserts the WaveSpeed
// text-to-image mixin advertises the shared Hugging Face text-to-image codec.
func TestWavespeedTextToImageMixin_ReusesHFCodec(t *testing.T) {
	t.Parallel()

	codec := WavespeedTextToImageMixin{}.TextToImageCodec()

	body, headers, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
		Context: context.Background(),
		Request: hftypes.TextToImageRequest{
			Input: "a cat",
			Parameters: &hftypes.TextToImageParameters{
				Width: new(512),
			},
		},
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"inputs":"a cat","parameters":{"width":512}}`, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "*/*", headers.Get("Accept"))
}
