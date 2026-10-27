package hfproviders

import (
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

const wavespeedProviderName = "wavespeed"

// WavespeedProvider serves text-to-video, text-to-image, and image-to-image
// tasks through the Hugging Face router using WaveSpeed's partner model
// identifiers. Callers configure the Hugging Face model ID they know; each
// embedded mixin translates it to WaveSpeed's identifier at
// endpoint-construction time via its [ModelResolver]. The Hugging Face router
// is a path-transparent proxy, so the endpoint path embeds the provider slug
// and WaveSpeed's own model ID.
//
// The zero value is safe to use: endpoint methods return a configuration
// [hferrors.SDKError] instead of dereferencing a nil resolver. Embedding all
// three mixins means the promoted field WavespeedProvider.Resolver is
// ambiguous — access each mixin's resolver explicitly if you ever need to
// (e.g. p.WavespeedTextToImageMixin.Resolver).
//
// As with the Hugging Face mixins, each WaveSpeed mixin can be embedded on its
// own in a custom provider that also implements Provider.Name and
// Provider.ProviderSuffix.
type WavespeedProvider struct {
	WavespeedTextToVideoMixin
	WavespeedTextToImageMixin
	WavespeedImageToImageMixin
}

// Compile-time assertions that WavespeedProvider satisfies each supported
// per-task provider interface.
var (
	_ ImageToImageProvider = (*WavespeedProvider)(nil)
	_ TextToImageProvider  = (*WavespeedProvider)(nil)
	_ TextToVideoProvider  = (*WavespeedProvider)(nil)
)

// NewWavespeedProvider creates a [WavespeedProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value WavespeedProvider is still safe to call; its
// endpoint methods return a configuration [hferrors.SDKError].
func NewWavespeedProvider(resolver ModelResolver) (WavespeedProvider, error) {
	if resolver == nil {
		return WavespeedProvider{}, errNilResolver(wavespeedProviderName)
	}

	return WavespeedProvider{
		WavespeedTextToVideoMixin: WavespeedTextToVideoMixin{Resolver: resolver},
		WavespeedTextToImageMixin: WavespeedTextToImageMixin{
			HuggingFaceTextToImageMixin: HuggingFaceTextToImageMixin{},
			Resolver:                    resolver,
		},
		WavespeedImageToImageMixin: WavespeedImageToImageMixin{Resolver: resolver},
	}, nil
}

// Name returns the name of the WaveSpeed provider for debug purposes.
func (p WavespeedProvider) Name() string {
	return wavespeedProviderName
}

// ProviderSuffix returns an empty string. WaveSpeed routes via the endpoint
// path, not a model suffix.
func (p WavespeedProvider) ProviderSuffix() string {
	return ""
}

// resolveWavespeedEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's provider model identifier and returns the router-proxied
// endpoint for the given task. The provider slug given to the resolver is
// WaveSpeed's Hub identifier — the key the Hub's mapping data uses — which is
// a provider fact, not something the SDK derives.
func resolveWavespeedEndpoint(
	resolver ModelResolver,
	params EndpointParams,
	task Task,
) (Endpoint, error) {
	if resolver == nil {
		return Endpoint{}, errNilResolver(wavespeedProviderName)
	}

	providerModelID, err := resolver.ResolveModel(params.Context, ResolveModelParams{
		Provider: wavespeedProviderName,
		Task:     task,
		Model:    params.Model,
	})
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{Method: http.MethodPost, Path: "wavespeed/api/v3/" + providerModelID}, nil
}

// WavespeedTextToVideoMixin provides the WaveSpeed endpoint and codec for
// text-to-video generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-video. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedTextToVideoMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToVideoEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedTextToVideoMixin) TextToVideoEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskTextToVideo)
}

// TextToVideoCodec returns the text-to-video codec, which encodes requests as
// JSON and returns the generated video as raw bytes.
func (WavespeedTextToVideoMixin) TextToVideoCodec() TextToVideoCodec {
	return wavespeedTextToVideoCodec{}
}

// WavespeedTextToImageMixin provides the WaveSpeed endpoint for text-to-image
// generation and reuses the Hugging Face text-to-image codec, whose wire
// format WaveSpeed shares: requests are encoded as JSON and the generated
// image is returned as raw bytes. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedTextToImageMixin struct {
	// HuggingFaceTextToImageMixin supplies the shared text-to-image codec.
	HuggingFaceTextToImageMixin

	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedTextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskTextToImage)
}

// WavespeedImageToImageMixin provides the WaveSpeed endpoint and codec for
// image-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// image-to-image. Its [ModelResolver] maps the configured Hugging Face model ID
// to WaveSpeed's provider model identifier at endpoint-construction time.
type WavespeedImageToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to WaveSpeed's
	// provider model identifier. It must be non-nil when the mixin's endpoint
	// method is called.
	Resolver ModelResolver
}

// ImageToImageEndpoint resolves the configured Hugging Face model ID to
// WaveSpeed's identifier and returns the router-proxied path for it.
func (m WavespeedImageToImageMixin) ImageToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	return resolveWavespeedEndpoint(m.Resolver, params, TaskImageToImage)
}

// ImageToImageCodec returns the image-to-image codec, which encodes requests as
// JSON and returns the generated image as raw bytes.
func (WavespeedImageToImageMixin) ImageToImageCodec() ImageToImageCodec {
	return wavespeedImageToImageCodec{}
}

// wavespeedTextToVideoCodec is a [Codec] for text-to-video that encodes canonical
// requests as JSON and decodes the generated video from the raw response body.
// The exact video subtype is provider-chosen, so requests advertise
// "Accept: */*" and content-type validation accepts any video media type,
// returning the body plus the parsed media type.
type wavespeedTextToVideoCodec struct{}

func (wavespeedTextToVideoCodec) Encode(
	params EncodeParams[hftypes.TextToVideoRequest],
) (body []byte, headers http.Header, err error) {
	body, headers, err = JSONCodec[
		hftypes.TextToVideoRequest,
		hftypes.TextToVideoResponse,
	]{}.Encode(params)
	if err != nil {
		return nil, nil, err
	}
	headers.Set("Accept", "*/*")

	return body, headers, nil
}

func (wavespeedTextToVideoCodec) Decode(
	params DecodeParams,
) (hftypes.TextToVideoResponse, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, "video/") {
		return hftypes.TextToVideoResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected a video Content-Type, got " + contentType,
			Err:     nil,
		}
	}

	return hftypes.TextToVideoResponse{
		Video:     params.Body,
		MediaType: mediaType,
	}, nil
}

// wavespeedImageToImageCodec is a [Codec] for image-to-image that encodes
// canonical requests as JSON and decodes the generated image from the raw
// response body. The exact image subtype is provider-chosen, so requests
// advertise "Accept: */*" and content-type validation accepts any image media
// type, returning the body plus the parsed media type.
type wavespeedImageToImageCodec struct{}

func (wavespeedImageToImageCodec) Encode(
	params EncodeParams[hftypes.ImageToImageRequest],
) (body []byte, headers http.Header, err error) {
	body, headers, err = JSONCodec[
		hftypes.ImageToImageRequest,
		hftypes.ImageToImageResponse,
	]{}.Encode(params)
	if err != nil {
		return nil, nil, err
	}
	headers.Set("Accept", "*/*")

	return body, headers, nil
}

func (wavespeedImageToImageCodec) Decode(
	params DecodeParams,
) (hftypes.ImageToImageResponse, error) {
	contentType := params.Headers.Get("Content-Type")

	mediaType, ok := parseMediaType(contentType)
	if !ok || !strings.HasPrefix(mediaType, "image/") {
		return hftypes.ImageToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected an image Content-Type, got " + contentType,
			Err:     nil,
		}
	}

	return hftypes.ImageToImageResponse{
		Image:     params.Body,
		MediaType: mediaType,
	}, nil
}
