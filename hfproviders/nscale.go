package hfproviders

import (
	"encoding/base64"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// nscaleProviderName is the Nscale provider slug: the model-suffix token the
// chat task pins routing with, the router path segment text-to-image requests
// are proxied under, and the Hub identifier the model resolver looks mappings
// up by.
const nscaleProviderName = "nscale"

// nscaleTextToImagePath is the router-proxied path for Nscale's
// OpenAI-compatible image-generation endpoint, relative to the caller's base
// URL.
const nscaleTextToImagePath = nscaleProviderName + "/v1/images/generations"

// nscaleResponseFormatB64 is the response format the text-to-image codec
// pins: the base64 JSON shape is the only envelope the codec decodes, so a
// caller-supplied value cannot switch the exchange to URL responses the
// codec would then fail on.
const nscaleResponseFormatB64 = "b64_json"

// NscaleProvider serves chat completion and text-to-image tasks through the
// Hugging Face router.
//
// Chat (unary and streaming, including vision-language requests) reuses
// [HuggingFaceChatMixin] and pins routing by appending the router's
// ":nscale" suffix to the model ID, exactly as the other chat partner
// providers do: the router receives the Hugging Face model ID plus suffix and
// performs provider mapping server-side. The suffix is a deterministic pin —
// the router's automatic failover applies to unsuffixed models and routing
// policies (":fastest", ":cheapest", ":preferred"), not to an explicit
// provider — so a model Nscale does not serve fails rather than running on
// another provider. Pass a request model with its own suffix (e.g.
// "model:preferred") to opt out of the pin. This routes through the unified
// OpenAI-compatible chat endpoint rather than the provider-scoped path the
// official Hugging Face clients use for Nscale, matching the convention
// shared by every other chat partner provider in this package.
//
// Text-to-image is served through Nscale's OpenAI-compatible images endpoint,
// proxied by the router under the "nscale" path prefix, with Nscale's own
// model identifier in the request body. Callers configure the Hugging Face
// model ID they know; the embedded mixin translates it to Nscale's identifier
// at endpoint-construction time via its [ModelResolver], and the codec reuses
// the resolution from [EncodeParams.Mapping]. The wire format mirrors
// huggingface.js: canonical parameters are forwarded as-is — width/height are
// NOT folded into a "size" field and num_inference_steps is NOT dropped,
// unlike the official Python client — ProviderParams merge over the typed
// fields, and response_format, prompt, and the mapped model are applied last
// so ProviderParams entries cannot override them. Responses carry a
// base64-encoded image in data[0].b64_json; the codec decodes it, reports the
// media type sniffed from the image bytes, and exposes the response's
// "created" timestamp on ProviderMetadata when present. Nscale-specific
// request parameters such as "n" or "size" ride the request's ProviderParams
// field.
//
// The zero value is safe to use: chat works without any resolver, while the
// text-to-image endpoint methods return a configuration [hferrors.SDKError]
// instead of dereferencing a nil resolver. As with the Hugging Face mixins,
// the Nscale mixin can be embedded on its own in a custom provider that also
// implements Provider.Name and Provider.ProviderSuffix.
type NscaleProvider struct {
	HuggingFaceChatMixin
	NscaleTextToImageMixin
}

// Compile-time assertions that NscaleProvider satisfies each supported
// per-task provider interface.
var (
	_ ChatProvider        = (*NscaleProvider)(nil)
	_ ChatStreamProvider  = (*NscaleProvider)(nil)
	_ TextToImageProvider = (*NscaleProvider)(nil)
)

// NewNscaleProvider creates an [NscaleProvider] backed by the given model
// resolver, e.g. a [HubModelResolver] for live Hub mappings or a
// [StaticModelResolver] for pinned deployments. It returns an error when
// resolver is nil so callers cannot accidentally create a provider that fails
// at request time. A zero-value NscaleProvider is still safe to call; its
// text-to-image endpoint method returns a configuration [hferrors.SDKError].
func NewNscaleProvider(resolver ModelResolver) (NscaleProvider, error) {
	if resolver == nil {
		return NscaleProvider{}, errNilResolver(nscaleProviderName)
	}

	return NscaleProvider{
		HuggingFaceChatMixin: HuggingFaceChatMixin{},
		Resolver:             resolver,
	}, nil
}

// Name returns the name of the Nscale provider for debug purposes.
func (p NscaleProvider) Name() string {
	return nscaleProviderName
}

// ProviderSuffix returns the "nscale" provider suffix for appending to the
// model selection in chat completion tasks. Text-to-image routes via its
// endpoint path and does not use the suffix.
func (p NscaleProvider) ProviderSuffix() string {
	return nscaleProviderName
}

// NscaleTextToImageMixin provides the Nscale endpoint and codec for
// text-to-image generation. It is a mixin: embed it in a type that also
// implements Provider.Name and Provider.ProviderSuffix to serve only
// text-to-image. Its [ModelResolver] maps the configured Hugging Face model
// ID to Nscale's provider model identifier at endpoint-construction time.
type NscaleTextToImageMixin struct {
	// Resolver maps the configured Hugging Face model ID to Nscale's
	// provider model identifier. It must be non-nil when the mixin's
	// endpoint method is called.
	Resolver ModelResolver
}

// TextToImageEndpoint resolves the configured Hugging Face model ID to
// Nscale's identifier and returns the router-proxied path for the
// image-generation endpoint.
func (m NscaleTextToImageMixin) TextToImageEndpoint(
	params EndpointParams,
) (Endpoint, error) {
	mapping, err := resolveMapping(
		params.Context,
		nscaleProviderName,
		m.Resolver,
		nil,
		TaskTextToImage,
		params.Model,
	)
	if err != nil {
		return Endpoint{}, err
	}

	return Endpoint{
		Method:  http.MethodPost,
		Path:    nscaleTextToImagePath,
		Mapping: mapping,
	}, nil
}

// TextToImageCodec returns the text-to-image codec, which encodes requests
// as Nscale's OpenAI-style images JSON body and decodes the base64-encoded
// response, reusing the mapping resolved at endpoint-construction time and
// falling back to the mixin's resolver when used standalone.
func (m NscaleTextToImageMixin) TextToImageCodec() TextToImageCodec {
	return nscaleTextToImageCodec{resolver: m.Resolver}
}

// nscaleImageGenerationData is one generated image in Nscale's response. The
// request pins response_format to b64_json, so the base64 field is the
// expected carrier of the image bytes.
type nscaleImageGenerationData struct {
	B64JSON string `json:"b64_json"`
}

// nscaleImageGeneration is the OpenAI-style response envelope Nscale returns
// for text-to-image.
type nscaleImageGeneration struct {
	// Created is the Unix timestamp the image was generated at. Optional:
	// the codec only exposes it when the response carries it.
	Created *int64 `json:"created,omitempty"`

	Data []nscaleImageGenerationData `json:"data"`
}

// nscaleTextToImageCodec is a [Codec] for text-to-image that encodes
// canonical requests into Nscale's OpenAI-style images JSON body and decodes
// Nscale's base64 JSON response into the generated image bytes.
//
// The body mirrors huggingface.js's payload preparation: the canonical
// parameters are flattened as-is (see [hftypes.TextToImageParameters]),
// ProviderParams merge over them, and response_format, prompt, and the mapped
// provider model identifier are applied last so ProviderParams entries cannot
// override them. This is the only wire format Nscale's images endpoint
// accepts, unlike the Hugging Face provider's raw-bytes variant.
type nscaleTextToImageCodec struct {
	// resolver resolves the model mapping when EncodeParams.Mapping is nil,
	// i.e. when the codec is used standalone. Nil makes Encode fail with a
	// configuration error unless a mapping is supplied.
	resolver ModelResolver
}

// Compile-time assertion that the codec implements the text-to-image task
// codec.
var _ TextToImageCodec = (*nscaleTextToImageCodec)(nil)

func (c nscaleTextToImageCodec) Encode(
	params EncodeParams[hftypes.TextToImageRequest],
) (body []byte, headers http.Header, err error) {
	req := params.Request

	mapping, err := resolveMapping(
		params.Context,
		nscaleProviderName,
		c.resolver,
		params.Mapping,
		TaskTextToImage,
		params.Model,
	)
	if err != nil {
		return nil, nil, err
	}

	// Nscale's body is flat: the prompt, the mapped model, and every
	// generation parameter live at the top level.
	payload := map[string]any{}
	if err := mergeRequestParameters(payload, req.Parameters, req.ProviderParams); err != nil {
		return nil, nil, err
	}
	payload["response_format"] = nscaleResponseFormatB64
	payload["prompt"] = req.Input
	payload["model"] = mapping.ProviderModelID

	body, err = marshalRequestBody(payload)
	if err != nil {
		return nil, nil, err
	}

	return body, jsonRequestHeaders(mimeTypeJSON), nil
}

func (c nscaleTextToImageCodec) Decode(
	params DecodeParams,
) (hftypes.TextToImageResponse, error) {
	envelope, err := JSONCodec[hftypes.TextToImageRequest, nscaleImageGeneration]{}.Decode(params)
	if err != nil {
		return hftypes.TextToImageResponse{}, err
	}
	if len(envelope.Data) == 0 || envelope.Data[0].B64JSON == "" {
		return hftypes.TextToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "Nscale image response contains no image data",
			Err:     nil,
		}
	}

	image, err := base64.StdEncoding.DecodeString(envelope.Data[0].B64JSON)
	if err != nil {
		return hftypes.TextToImageResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode base64 image data",
			Err:     err,
		}
	}

	// The envelope is JSON, so the image's media type is not on a header; it
	// is sniffed from the decoded bytes. A payload that is not an image
	// family is rejected rather than surfaced as response MediaType of
	// "application/octet-stream" or worse.
	mediaType := http.DetectContentType(image)
	if !strings.HasPrefix(mediaType, "image/") {
		return hftypes.TextToImageResponse{}, errUnexpectedMediaPrefix("image/", mediaType)
	}

	var metadata map[string]any
	if envelope.Created != nil {
		metadata = map[string]any{"created": *envelope.Created}
	}

	return hftypes.TextToImageResponse{
		Image:            image,
		MediaType:        mediaType,
		ProviderMetadata: metadata,
	}, nil
}
