//go:build !integration

package hfproviders

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const nscaleTestModel = "black-forest-labs/FLUX.1-schnell"

// nscaleRecordingResolver records the ResolveModel lookup key and serves a
// fixed mapping, letting endpoint tests assert what the provider looked up.
type nscaleRecordingResolver struct {
	got     ResolveModelParams
	mapping ModelMapping
}

// ResolveModel implements [ModelResolver].
func (r *nscaleRecordingResolver) ResolveModel(
	_ context.Context,
	params ResolveModelParams,
) (ModelMapping, error) {
	r.got = params

	return r.mapping, nil
}

// nscaleFailingResolver resolves nothing and always fails, emulating a
// resolver outage that endpoints and codecs must propagate.
type nscaleFailingResolver struct{}

// ResolveModel implements [ModelResolver].
func (nscaleFailingResolver) ResolveModel(
	context.Context,
	ResolveModelParams,
) (ModelMapping, error) {
	return ModelMapping{}, errors.New("resolver outage")
}

// newNscaleProvider builds an NscaleProvider backed by a static resolver
// covering the test model.
func newNscaleProvider(t *testing.T) NscaleProvider {
	t.Helper()

	provider, err := NewNscaleProvider(NewStaticModelResolver(map[StaticMapping]ModelMapping{
		{
			Provider: nscaleProviderName,
			Task:     TaskTextToImage,
			Model:    nscaleTestModel,
		}: {ProviderModelID: nscaleTestModel},
	}))
	require.NoError(t, err)

	return provider
}

// nscaleTestMapping is the mapping endpoint construction supplies in pipeline
// use; encode tests reuse it so the codec never needs a resolver.
func nscaleTestMapping() *ModelMapping {
	return &ModelMapping{ProviderModelID: nscaleTestModel}
}

func TestNscaleProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	provider := newNscaleProvider(t)

	require.Equal(t, "nscale", provider.Name())
	require.Equal(t, "nscale", provider.ProviderSuffix())
}

func TestNewNscaleProvider_NilResolver(t *testing.T) {
	t.Parallel()

	provider, err := NewNscaleProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, NscaleProvider{}, provider)
}

// TestNscaleProvider_ZeroValueIsSafe verifies the zero-value provider's
// text-to-image endpoint and codec fail with configuration errors instead of
// dereferencing a nil resolver.
func TestNscaleProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var provider NscaleProvider

	_, err := provider.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, _, err = provider.TextToImageCodec().Encode(EncodeParams[hftypes.TextToImageRequest]{
		Context: context.Background(),
		Request: hftypes.TextToImageRequest{Input: "a cat"},
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestNscaleProvider_Contract pins the task interfaces NscaleProvider serves
// and the ones it must reject.
func TestNscaleProvider_Contract(t *testing.T) {
	t.Parallel()

	provider := newNscaleProvider(t)

	for _, served := range []struct {
		typeName string
		cast     func(Provider) error
	}{
		{
			typeName: "hfproviders.ChatProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[ChatProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.ChatStreamProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[ChatStreamProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TextToImageProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextToImageProvider](p)

				return err
			},
		},
	} {
		require.NoError(t, served.cast(provider), "must implement %s", served.typeName)
	}

	for _, missing := range []struct {
		typeName string
		cast     func(Provider) error
	}{
		{
			typeName: "hfproviders.SummarizationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SummarizationProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TextClassificationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextClassificationProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.SpeechRecognitionProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SpeechRecognitionProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.FeatureExtractionPooledProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[FeatureExtractionPooledProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.ImageToImageProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[ImageToImageProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TextToVideoProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextToVideoProvider](p)

				return err
			},
		},
	} {
		err := missing.cast(provider)
		require.Error(t, err, "must not implement %s", missing.typeName)
		require.ErrorContains(t, err, "nscale does not implement "+missing.typeName)
	}
}

//nolint:dupl // Chat-endpoint tests intentionally mirror each other; each provider documents its own endpoint wiring.
func TestNscaleProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	provider := newNscaleProvider(t)

	chat, err := AsProvider[ChatProvider](provider)
	require.NoError(t, err)
	stream, err := AsProvider[ChatStreamProvider](provider)
	require.NoError(t, err)

	// Chat routes through the fixed OpenAI-compatible path, model-in-body:
	// an empty Model is legal and no mapping is attached.
	endpoint, err := chat.ChatEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "meta-llama/Llama-3.1-8B-Instruct:nscale",
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
	require.Nil(t, endpoint.Mapping)

	endpoint, err = chat.ChatEndpoint(EndpointParams{Context: context.Background()})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)

	// Codec wiring: the built-in JSON codecs, pinning that chat reuses the
	// default OpenAI-compatible wire format.
	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
		chat.ChatCodec(),
	)
	require.IsType(
		t,
		JSONCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		stream.ChatStreamCodec(),
	)
}

func TestNscaleProvider_TextToImageEndpoint(t *testing.T) {
	t.Parallel()

	resolver := &nscaleRecordingResolver{
		mapping: ModelMapping{
			ProviderModelID: nscaleTestModel,
			Status:          "live",
		},
	}
	provider, err := NewNscaleProvider(resolver)
	require.NoError(t, err)

	textToImage, err := AsProvider[TextToImageProvider](provider)
	require.NoError(t, err)

	endpoint, err := textToImage.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   nscaleTestModel,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "nscale/v1/images/generations", endpoint.Path)
	require.Equal(t, nscaleTestModel, endpoint.Mapping.ProviderModelID)

	require.Equal(t, ResolveModelParams{
		Provider: "nscale",
		Task:     TaskTextToImage,
		Model:    nscaleTestModel,
	}, resolver.got)
}

// TestNscaleProvider_TextToImageEndpoint_ResolutionError pins that an
// unresolvable model surfaces as a configuration error from the endpoint.
func TestNscaleProvider_TextToImageEndpoint_ResolutionError(t *testing.T) {
	t.Parallel()

	provider, err := NewNscaleProvider(NewStaticModelResolver(nil))
	require.NoError(t, err)

	textToImage, err := AsProvider[TextToImageProvider](provider)
	require.NoError(t, err)

	_, err = textToImage.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "unknown-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// nscaleRequireFlatEncode asserts an encoder produced the expected flat
// Nscale body with JSON content headers.
func nscaleRequireFlatEncode(
	t *testing.T,
	encode func(EncodeParams[hftypes.TextToImageRequest]) ([]byte, http.Header, error),
	request hftypes.TextToImageRequest,
	wantJSON string,
) {
	t.Helper()

	body, headers, err := encode(EncodeParams[hftypes.TextToImageRequest]{
		Context: context.Background(),
		Request: request,
		Mapping: nscaleTestMapping(),
	})
	require.NoError(t, err)
	require.JSONEq(t, wantJSON, string(body))
	require.Equal(t, "application/json", headers.Get("Content-Type"))
	require.Equal(t, "application/json", headers.Get("Accept"))
}

func TestNscaleTextToImageCodec_Encode(t *testing.T) {
	t.Parallel()

	codec := nscaleTextToImageCodec{}

	t.Run("nil parameters yields prompt, model, and response format only", func(t *testing.T) {
		t.Parallel()

		nscaleRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{Input: "a cat"},
			`{"response_format":"b64_json","prompt":"a cat","model":"`+nscaleTestModel+`"}`,
		)
	})

	t.Run("canonical parameters are forwarded as-is", func(t *testing.T) {
		t.Parallel()

		// Mirrors huggingface.js: width/height stay flat (no size fold) and
		// num_inference_steps is not dropped, unlike the official Python
		// client.
		nscaleRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					GuidanceScale:     new(7.5),
					NegativePrompt:    new("no dogs"),
					NumInferenceSteps: new(30),
					Width:             new(512),
					Height:            new(768),
					Scheduler:         new("euler"),
					Seed:              new(int64(42)),
				},
			},
			`{"guidance_scale":7.5,"negative_prompt":"no dogs","num_inference_steps":30,`+
				`"width":512,"height":768,"scheduler":"euler","seed":42,`+
				`"response_format":"b64_json","prompt":"a cat","model":"`+nscaleTestModel+`"}`,
		)
	})

	t.Run("provider parameters merge over typed fields", func(t *testing.T) {
		t.Parallel()

		nscaleRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				Parameters: &hftypes.TextToImageParameters{
					Seed: new(int64(42)),
				},
				ProviderParams: map[string]any{
					"seed": 99,
					"n":    2,
				},
			},
			`{"seed":99,"n":2,"response_format":"b64_json","prompt":"a cat","model":"`+nscaleTestModel+`"}`,
		)
	})

	t.Run("protected fields win over provider parameters", func(t *testing.T) {
		t.Parallel()

		nscaleRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input: "a cat",
				ProviderParams: map[string]any{
					"prompt":          "attacker prompt",
					"model":           "attacker-model",
					"response_format": "url",
				},
			},
			`{"response_format":"b64_json","prompt":"a cat","model":"`+nscaleTestModel+`"}`,
		)
	})

	t.Run("nil provider parameters ride through as JSON null", func(t *testing.T) {
		// Mirrors the other flat-JSON providers (WaveSpeed, fal.ai), whose
		// maps.Copy merge passes nil-valued entries through to the body.
		t.Parallel()

		nscaleRequireFlatEncode(
			t,
			codec.Encode,
			hftypes.TextToImageRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"ignored": nil},
			},
			`{"ignored":null,"response_format":"b64_json","prompt":"a cat","model":"`+nscaleTestModel+`"}`,
		)
	})

	t.Run("unmarshalable provider parameter fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{
				Input:          "a cat",
				ProviderParams: map[string]any{"bad": make(chan int)},
			},
			Model:   nscaleTestModel,
			Mapping: nscaleTestMapping(),
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("supplied mapping wins without a resolver call", func(t *testing.T) {
		t.Parallel()

		resolver := &countingResolver{}
		codec := NscaleTextToImageMixin{Resolver: resolver}.TextToImageCodec()

		body, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   nscaleTestModel,
			Mapping: &ModelMapping{ProviderModelID: "mapped-at-endpoint-construction"},
		})
		require.NoError(t, err)
		require.Zero(t, resolver.calls.Load())
		require.JSONEq(
			t,
			`{"response_format":"b64_json","prompt":"a cat","model":"mapped-at-endpoint-construction"}`,
			string(body),
		)
	})

	t.Run("missing mapping and resolver fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   nscaleTestModel,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("resolver errors propagate", func(t *testing.T) {
		t.Parallel()

		codec := NscaleTextToImageMixin{Resolver: nscaleFailingResolver{}}.TextToImageCodec()

		_, _, err := codec.Encode(EncodeParams[hftypes.TextToImageRequest]{
			Context: context.Background(),
			Request: hftypes.TextToImageRequest{Input: "a cat"},
			Model:   nscaleTestModel,
		})
		require.ErrorContains(t, err, "resolver outage")
	})
}

func TestNscaleTextToImageCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := nscaleTextToImageCodec{}

	png, err := base64.StdEncoding.DecodeString(testutils.TinyPNGBase64)
	require.NoError(t, err)

	jsonHeaders := func() http.Header {
		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		return headers
	}

	t.Run("first image decodes with sniffed media type", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body: []byte(`{"created":123,"data":[` +
				`{"b64_json":"` + testutils.TinyPNGBase64 + `"},` +
				`{"b64_json":"bm90LXVzZWQ="}]}`),
			Headers: jsonHeaders(),
		})
		require.NoError(t, err)
		require.Equal(t, png, resp.Image)
		require.Equal(t, "image/png", resp.MediaType)
		require.NotNil(t, resp.ProviderMetadata)
		require.Equal(t, int64(123), resp.ProviderMetadata["created"])
	})

	t.Run("response without created yields nil metadata", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"data":[{"b64_json":"` + testutils.TinyPNGBase64 + `"}]}`),
			Headers: jsonHeaders(),
		})
		require.NoError(t, err)
		require.Equal(t, png, resp.Image)
		require.Nil(t, resp.ProviderMetadata)
	})

	t.Run("empty data fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"created":123,"data":[]}`),
			Headers: jsonHeaders(),
		})
		require.ErrorContains(t, err, "contains no image data")

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
	})

	t.Run("empty b64_json fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"data":[{"b64_json":""}]}`),
			Headers: jsonHeaders(),
		})
		require.ErrorContains(t, err, "contains no image data")
	})

	t.Run("malformed base64 fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"data":[{"b64_json":"not base64!!"}]}`),
			Headers: jsonHeaders(),
		})
		require.ErrorContains(t, err, "failed to decode base64 image data")
	})

	t.Run("wrong content type fails", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "text/plain")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"data":[{"b64_json":"` + testutils.TinyPNGBase64 + `"}]}`),
			Headers: headers,
		})
		require.ErrorContains(t, err, "expected Content-Type application/json")
	})

	t.Run("malformed body fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte("{not json"),
			Headers: jsonHeaders(),
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// nscaleSubsetProvider serves only text-to-image by embedding the Nscale
// mixin with a resolver set directly, without going through NewNscaleProvider.
type nscaleSubsetProvider struct {
	NscaleTextToImageMixin
}

func (nscaleSubsetProvider) ProviderSuffix() string { return nscaleProviderName }

func (nscaleSubsetProvider) Name() string { return "subset" }

func TestNscaleMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	provider := nscaleSubsetProvider{
		NscaleTextToImageMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: nscaleProviderName,
					Task:     TaskTextToImage,
					Model:    nscaleTestModel,
				}: {ProviderModelID: nscaleTestModel},
			}),
		},
	}

	textToImage, err := AsProvider[TextToImageProvider](provider)
	require.NoError(t, err)

	endpoint, err := textToImage.TextToImageEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   nscaleTestModel,
	})
	require.NoError(t, err)
	require.Equal(t, "nscale/v1/images/generations", endpoint.Path)

	_, err = AsProvider[ChatProvider](provider)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.ChatProvider",
	)
}
