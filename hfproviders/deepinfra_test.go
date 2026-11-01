//go:build !integration

package hfproviders

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const (
	deepInfraASRModel   = "openai/whisper-large-v3"
	deepInfraEmbedModel = "BAAI/bge-base-en-v1.5"

	// deepInfraWAVAudio is a minimal RIFF/WAVE payload whose bytes net/http
	// sniffs as "audio/wave".
	deepInfraWAVAudio = "RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"

	// deepInfraMP3Audio is an ID3-tagged payload net/http sniffs as
	// "audio/mpeg".
	deepInfraMP3Audio = "ID3\x03\x00\x00\x00\x00\x00\x00"
)

// deepInfraRecordingResolver records the ResolveModel lookup key and serves a
// fixed mapping, letting endpoint tests assert what the provider looked up.
type deepInfraRecordingResolver struct {
	got     ResolveModelParams
	mapping ModelMapping
}

// ResolveModel implements [ModelResolver].
func (r *deepInfraRecordingResolver) ResolveModel(
	_ context.Context,
	params ResolveModelParams,
) (ModelMapping, error) {
	r.got = params

	return r.mapping, nil
}

// newDeepInfraProvider builds a DeepInfraProvider backed by a static resolver
// covering the test models.
func newDeepInfraProvider(t *testing.T) DeepInfraProvider {
	t.Helper()

	provider, err := NewDeepInfraProvider(NewStaticModelResolver(map[StaticMapping]ModelMapping{
		{
			Provider: deepinfraProviderName,
			Task:     TaskAutomaticSpeechRecognition,
			Model:    deepInfraASRModel,
		}: {ProviderModelID: deepInfraASRModel},
		{
			Provider: deepinfraProviderName,
			Task:     TaskFeatureExtraction,
			Model:    deepInfraEmbedModel,
		}: {ProviderModelID: deepInfraEmbedModel},
	}))
	require.NoError(t, err)

	return provider
}

// TestMediaSniffingForDeepInfra pins the net/http byte-sniffing behavior the
// DeepInfra speech-recognition codec relies on to pick the file part's
// Content-Type and filename extension. If a Go release changes these
// signatures, openAIAudioExtension needs revisiting.
func TestMediaSniffingForDeepInfra(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		wantMediaType string
		audio         []byte
	}{
		{
			name:          "wav",
			wantMediaType: "audio/wave",
			audio:         []byte(deepInfraWAVAudio),
		},
		{
			name:          "mp3",
			wantMediaType: "audio/mpeg",
			audio:         []byte(deepInfraMP3Audio),
		},
		{
			// MP4/M4A containers: net/http labels them video/mp4 when the
			// ftyp brands contain "mp4", regardless of audio content.
			name:          "mp4 container",
			wantMediaType: "video/mp4",
			audio:         []byte("\x00\x00\x00\x18ftypisom\x00\x00\x02\x00mp42isom"),
		},
		{
			name:          "webm",
			wantMediaType: "video/webm",
			audio:         []byte{0x1A, 0x45, 0xDF, 0xA3},
		},
		{
			name:          "ogg",
			wantMediaType: "application/ogg",
			audio:         []byte("OggS\x00\x02\x00\x00"),
		},
		{
			// net/http has no AMR signature either, and AMR files are
			// ASCII-ish: the sniffer reports text/plain, so AMR input falls
			// back to the ".wav" filename like any unrecognized format.
			name:          "amr (net/http undetected)",
			wantMediaType: "text/plain; charset=utf-8",
			audio:         []byte("#!AMR\n"),
		},
		{
			// net/http has no FLAC signature: the sniffer reports
			// application/octet-stream, and the speech-recognition codec
			// checks the "fLaC" prefix explicitly to label FLAC input.
			name:          "flac (net/http undetected)",
			wantMediaType: "application/octet-stream",
			audio:         []byte("fLaC\x00\x00\x00\x22"),
		},
		{
			name:          "unknown binary",
			wantMediaType: "application/octet-stream",
			audio:         []byte{0x00, 0x01, 0x02, 0x03},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.wantMediaType, http.DetectContentType(tc.audio))
		})
	}
}

func TestDeepInfraProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)

	require.Equal(t, "deepinfra", provider.Name())
	require.Equal(t, "deepinfra", provider.ProviderSuffix())
}

func TestNewDeepInfraProvider_NilResolver(t *testing.T) {
	t.Parallel()

	provider, err := NewDeepInfraProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, DeepInfraProvider{}, provider)
}

// TestDeepInfraProvider_ZeroValueIsSafe verifies the zero-value provider's
// pipeline endpoint methods and codecs fail with configuration errors
// instead of dereferencing a nil resolver.
func TestDeepInfraProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var provider DeepInfraProvider

	_, err := provider.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = provider.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, err = provider.FeatureExtractionBatchEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	var (
		speechMixin  DeepInfraSpeechRecognitionMixin
		featureMixin DeepInfraFeatureExtractionMixin
	)

	_, _, err = speechMixin.SpeechRecognitionCodec().Encode(
		EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, _, err = featureMixin.FeatureExtractionPooledCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, _, err = featureMixin.FeatureExtractionPooledBatchCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionBatchRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

// TestDeepInfraProvider_Contract pins the task interfaces DeepInfraProvider
// serves and the ones it must reject.
func TestDeepInfraProvider_Contract(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)

	requireServedTaskInterfaces(t, provider, []func(Provider) error{
		func(p Provider) error {
			_, err := AsProvider[ChatProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[ChatStreamProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[SpeechRecognitionProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[FeatureExtractionPooledProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[FeatureExtractionPooledBatchProvider](p)

			return err
		},
	})

	requireMissingTaskInterfaces(t, provider, "deepinfra", []taskCast{
		{
			typeName: "hfproviders.SummarizationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SummarizationProvider](p)

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
		{
			typeName: "hfproviders.FeatureExtractionTokenProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[FeatureExtractionTokenProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.FeatureExtractionTokenBatchProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[FeatureExtractionTokenBatchProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TokenClassificationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TokenClassificationProvider](p)

				return err
			},
		},
	})
}

func TestDeepInfraProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)

	requireOpenAIChatPartnerEndpoint(t, provider, "deepinfra")
}

func TestDeepInfraProvider_SpeechRecognitionEndpoint(t *testing.T) {
	t.Parallel()

	resolver := &deepInfraRecordingResolver{
		mapping: ModelMapping{
			ProviderModelID: "openai/whisper-large-v3-turbo",
			Status:          "live",
		},
	}
	provider, err := NewDeepInfraProvider(resolver)
	require.NoError(t, err)

	speech, err := AsProvider[SpeechRecognitionProvider](provider)
	require.NoError(t, err)

	endpoint, err := speech.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   deepInfraASRModel,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "deepinfra/v1/openai/audio/transcriptions", endpoint.Path)
	require.Equal(t, "openai/whisper-large-v3-turbo", endpoint.Mapping.ProviderModelID)

	require.Equal(t, ResolveModelParams{
		Provider: "deepinfra",
		Task:     TaskAutomaticSpeechRecognition,
		Model:    deepInfraASRModel,
	}, resolver.got)
}

func TestDeepInfraProvider_FeatureExtractionEndpoints(t *testing.T) {
	t.Parallel()

	resolver := &deepInfraRecordingResolver{
		mapping: ModelMapping{ProviderModelID: "BAAI/bge-base-en-v1.5", Status: "live"},
	}
	provider, err := NewDeepInfraProvider(resolver)
	require.NoError(t, err)

	requireOpenAIEmbeddingsEndpoints(
		t,
		provider,
		deepInfraEmbedModel,
		"deepinfra/v1/openai/embeddings",
		"BAAI/bge-base-en-v1.5",
	)

	require.Equal(t, ResolveModelParams{
		Provider: "deepinfra",
		Task:     TaskFeatureExtraction,
		Model:    deepInfraEmbedModel,
	}, resolver.got)
}

// deepInfraMultipartPart holds one decoded multipart part.
type deepInfraMultipartPart struct {
	formName    string
	filename    string
	contentType string
	value       []byte
}

// deepInfraParseMultipart splits an encoded multipart body, using the
// boundary carried by the given Content-Type header.
func deepInfraParseMultipart(
	t *testing.T,
	contentType string,
	body []byte,
) []deepInfraMultipartPart {
	t.Helper()

	mediaType, params, err := mime.ParseMediaType(contentType)
	require.NoError(t, err)
	require.Equal(t, "multipart/form-data", mediaType)

	boundary, ok := params["boundary"]
	require.True(t, ok, "content type must carry a boundary")

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var parts []deepInfraMultipartPart
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return parts
		}
		require.NoError(t, err)

		value, err := io.ReadAll(part)
		require.NoError(t, err)
		require.NoError(t, part.Close())

		parts = append(parts, deepInfraMultipartPart{
			formName:    part.FormName(),
			filename:    part.FileName(),
			contentType: part.Header.Get("Content-Type"),
			value:       value,
		})
	}
}

// deepInfraFindPart returns the part with the given form name.
func deepInfraFindPart(
	t *testing.T,
	parts []deepInfraMultipartPart,
	formName string,
) deepInfraMultipartPart {
	t.Helper()

	for _, part := range parts {
		if part.formName == formName {
			return part
		}
	}

	require.FailNow(t, "multipart part not found", "no part named %q", formName)

	return deepInfraMultipartPart{}
}

func TestDeepInfraSpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)
	codec := provider.SpeechRecognitionCodec()

	t.Run("multipart body carries audio, parameters, and model", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
				ProviderParams: map[string]any{"beam_size": 5},
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		contentType := headers.Get("Content-Type")
		require.True(
			t,
			strings.HasPrefix(contentType, "multipart/form-data; boundary="),
			"unexpected content type: %s", contentType,
		)
		require.Equal(t, "application/json", headers.Get("Accept"))

		parts := deepInfraParseMultipart(t, contentType, body)
		require.Len(t, parts, 4)

		file := deepInfraFindPart(t, parts, "file")
		require.Equal(t, "audio.wav", file.filename)
		require.Equal(t, "audio/wave", file.contentType)
		require.Equal(t, []byte(deepInfraWAVAudio), file.value)

		require.Equal(t, "true", string(deepInfraFindPart(t, parts, "return_timestamps").value))
		require.Equal(t, "5", string(deepInfraFindPart(t, parts, "beam_size").value))
		require.Equal(t, deepInfraASRModel, string(deepInfraFindPart(t, parts, "model").value))
	})

	t.Run("filename follows the sniffed media type", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraMP3Audio)),
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		file := deepInfraFindPart(t, parts, "file")
		require.Equal(t, "audio.mp3", file.filename)
		require.Equal(t, "audio/mpeg", file.contentType)

		// No parameters: only the file part and the model field.
		require.Len(t, parts, 2)
	})

	t.Run("provider parameters cannot override the model", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				ProviderParams: map[string]any{"model": "attacker-model"},
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.Equal(t, deepInfraASRModel, string(deepInfraFindPart(t, parts, "model").value))
	})

	t.Run("complex provider parameters JSON-encode", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				ProviderParams: map[string]any{
					"options": map[string]any{"a": 1},
					"strict":  true,
				},
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.JSONEq(t, `{"a":1}`, string(deepInfraFindPart(t, parts, "options").value))
		require.Equal(t, "true", string(deepInfraFindPart(t, parts, "strict").value))
	})

	t.Run("flac input is labeled flac", func(t *testing.T) {
		t.Parallel()

		// jfk.flac-style input: net/http cannot sniff FLAC, so the codec's
		// explicit "fLaC" check supplies the media type and extension.
		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString(
					append([]byte("fLaC"), make([]byte, 8)...),
				),
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		file := deepInfraFindPart(t, parts, "file")
		require.Equal(t, "audio.flac", file.filename)
		require.Equal(t, "audio/flac", file.contentType)
	})

	t.Run("unmarshalable provider parameter fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				ProviderParams: map[string]any{"bad": make(chan int)},
			},
			Model: deepInfraASRModel,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("nil provider parameters are skipped", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				ProviderParams: map[string]any{"ignored": nil},
			},
			Model: deepInfraASRModel,
		})
		require.NoError(t, err)

		// Only the file part and the model field survive the nil skip.
		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.Len(t, parts, 2)
	})

	t.Run("invalid base64 input fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{Input: "not base64!!"},
			Model:   deepInfraASRModel,
		})
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
		require.Contains(t, sdkErr.Message, "failed to decode base64 input")
	})

	t.Run("supplied mapping wins without a resolver call", func(t *testing.T) {
		t.Parallel()

		resolver := &countingResolver{}
		codec := DeepInfraSpeechRecognitionMixin{Resolver: resolver}.SpeechRecognitionCodec()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
			},
			Model: deepInfraASRModel,
			Mapping: &ModelMapping{
				ProviderModelID: "mapped-at-endpoint-construction",
			},
		})
		require.NoError(t, err)
		require.Zero(t, resolver.calls.Load())

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.Equal(
			t,
			"mapped-at-endpoint-construction",
			string(deepInfraFindPart(t, parts, "model").value),
		)
	})

	t.Run("missing mapping and resolver fails", func(t *testing.T) {
		t.Parallel()

		codec := DeepInfraSpeechRecognitionMixin{}.SpeechRecognitionCodec()

		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
			},
			Model: deepInfraASRModel,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

func TestDeepInfraSpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := DeepInfraSpeechRecognitionMixin{}.SpeechRecognitionCodec()

	jsonHeaders := func() http.Header {
		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		return headers
	}

	t.Run("transcript only", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"And my soul from out that shadow"}`),
			Headers: jsonHeaders(),
		})
		require.NoError(t, err)
		require.Equal(t, "And my soul from out that shadow", resp.Text)
		require.Empty(t, resp.Chunks)
		require.Nil(t, resp.ProviderMetadata)
	})

	t.Run("segments map to chunks and extras ride metadata", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body: []byte(`{
				"task": "transcribe",
				"language": "en",
				"duration": 11.3,
				"text": "hello world",
				"segments": [
					{"start": 0, "end": 4.5, "text": "hello"},
					{"start": 4.5, "end": 11.3, "text": "world"}
				],
				"usage": {"prompt_seconds": 1.5}
			}`),
			Headers: jsonHeaders(),
		})
		require.NoError(t, err)
		require.Equal(t, "hello world", resp.Text)

		require.Len(t, resp.Chunks, 2)
		require.Equal(t, "hello", resp.Chunks[0].Text)
		require.Equal(t, []float64{0, 4.5}, resp.Chunks[0].Timestamp)
		require.Equal(t, "world", resp.Chunks[1].Text)
		require.Equal(t, []float64{4.5, 11.3}, resp.Chunks[1].Timestamp)

		require.NotNil(t, resp.ProviderMetadata)
		require.Equal(t, "en", resp.ProviderMetadata["language"])
		require.InDelta(t, 11.3, resp.ProviderMetadata["duration"], 0.0001)
		require.Equal(
			t,
			map[string]any{"prompt_seconds": 1.5},
			resp.ProviderMetadata["usage"],
		)

		_, hasText := resp.ProviderMetadata["text"]
		require.False(t, hasText)
		_, hasSegments := resp.ProviderMetadata["segments"]
		require.False(t, hasSegments)
	})

	t.Run("null segments produce no chunks", func(t *testing.T) {
		t.Parallel()

		resp, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"hi","segments":null}`),
			Headers: jsonHeaders(),
		})
		require.NoError(t, err)
		require.Equal(t, "hi", resp.Text)
		require.Empty(t, resp.Chunks)
	})

	t.Run("missing text fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"segments":[]}`),
			Headers: jsonHeaders(),
		})
		require.Error(t, err)

		var sdkErr *hferrors.SDKError
		require.ErrorAs(t, err, &sdkErr)
		require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
		require.Contains(t, sdkErr.Message, "missing text")
	})

	t.Run("null text fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":null}`),
			Headers: jsonHeaders(),
		})
		require.ErrorContains(t, err, "missing text")
	})

	t.Run("non-string text fails", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":42}`),
			Headers: jsonHeaders(),
		})
		require.ErrorContains(t, err, "must be a string")
	})

	t.Run("non-array segments fail", func(t *testing.T) {
		t.Parallel()

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"hi","segments":{}}`),
			Headers: jsonHeaders(),
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("wrong content type fails", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "text/plain")

		_, err := codec.Decode(DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"text":"hi"}`),
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

func TestDeepInfraFeatureExtractionCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)

	assertOpenAIEmbeddingsCodecEncode(
		t,
		provider.FeatureExtractionPooledCodec(),
		deepInfraEmbedModel,
		deepInfraEmbedModel,
		func(resolver ModelResolver) FeatureExtractionPooledCodec {
			return DeepInfraFeatureExtractionMixin{
				Resolver: resolver,
			}.FeatureExtractionPooledCodec()
		},
	)
}

func TestDeepInfraFeatureExtractionBatchCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newDeepInfraProvider(t)
	codec := provider.FeatureExtractionPooledBatchCodec()

	body, _, err := codec.Encode(EncodeParams[hftypes.FeatureExtractionBatchRequest]{
		Context: context.Background(),
		Request: hftypes.FeatureExtractionBatchRequest{
			Inputs: []string{"first input", "second input"},
		},
		Model: deepInfraEmbedModel,
	})
	require.NoError(t, err)
	require.JSONEq(
		t,
		`{"input":["first input","second input"],"model":"BAAI/bge-base-en-v1.5"}`,
		string(body),
	)
}

// deepInfraSubsetProvider serves only feature extraction by embedding the
// corresponding DeepInfra mixin with a resolver set directly, without going
// through NewDeepInfraProvider.
type deepInfraSubsetProvider struct {
	DeepInfraFeatureExtractionMixin
}

func (deepInfraSubsetProvider) ProviderSuffix() string { return deepinfraProviderName }

func (deepInfraSubsetProvider) Name() string { return "subset" }

func TestDeepInfraMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	provider := deepInfraSubsetProvider{
		DeepInfraFeatureExtractionMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: deepinfraProviderName,
					Task:     TaskFeatureExtraction,
					Model:    deepInfraEmbedModel,
				}: {ProviderModelID: deepInfraEmbedModel},
			}),
		},
	}

	pooled, err := AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	endpoint, err := pooled.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   deepInfraEmbedModel,
	})
	require.NoError(t, err)
	require.Equal(t, "deepinfra/v1/openai/embeddings", endpoint.Path)

	_, err = AsProvider[SpeechRecognitionProvider](provider)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.SpeechRecognitionProvider",
	)

	_, err = AsProvider[ChatProvider](provider)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"subset does not implement hfproviders.ChatProvider",
	)
}

// TestDeepInfraProvider_Tasks is a compile-time pin that the
// constructor-built provider implements exactly the documented task
// interfaces; the casts run through AsProvider like the dispatch layer does.
func TestDeepInfraProvider_Tasks(t *testing.T) {
	t.Parallel()

	provider, err := NewDeepInfraProvider(NewStaticModelResolver(nil))
	require.NoError(t, err)

	_, err = AsProvider[SpeechRecognitionProvider](provider)
	require.NoError(t, err)

	_, err = AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	_, err = AsProvider[FeatureExtractionPooledBatchProvider](provider)
	require.NoError(t, err)
}

// deepInfraFailingResolver resolves nothing and always fails, emulating a
// resolver outage that endpoints and codecs must propagate.
type deepInfraFailingResolver struct{}

// ResolveModel implements [ModelResolver].
func (deepInfraFailingResolver) ResolveModel(
	context.Context,
	ResolveModelParams,
) (ModelMapping, error) {
	return ModelMapping{}, errors.New("resolver outage")
}

// TestDeepInfraEndpoint_ResolverErrorsPropagate pins that a resolver error
// (not just a nil resolver) surfaces from the endpoint methods.
func TestDeepInfraEndpoint_ResolverErrorsPropagate(t *testing.T) {
	t.Parallel()

	provider, err := NewDeepInfraProvider(deepInfraFailingResolver{})
	require.NoError(t, err)

	speech, err := AsProvider[SpeechRecognitionProvider](provider)
	require.NoError(t, err)

	_, err = speech.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   deepInfraASRModel,
	})
	require.ErrorContains(t, err, "resolver outage")

	feature, err := AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)

	_, err = feature.FeatureExtractionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   deepInfraEmbedModel,
	})
	require.ErrorContains(t, err, "resolver outage")

	batch, err := AsProvider[FeatureExtractionPooledBatchProvider](provider)
	require.NoError(t, err)

	_, err = batch.FeatureExtractionBatchEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   deepInfraEmbedModel,
	})
	require.ErrorContains(t, err, "resolver outage")
}

// TestDeepInfraCodec_ResolverErrorsPropagate pins that a resolver error
// surfaces from the codecs when used standalone (no mapping supplied).
func TestDeepInfraCodec_ResolverErrorsPropagate(t *testing.T) {
	t.Parallel()

	provider, err := NewDeepInfraProvider(deepInfraFailingResolver{})
	require.NoError(t, err)

	_, _, err = provider.SpeechRecognitionCodec().Encode(
		EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
			},
			Model: deepInfraASRModel,
		})
	require.ErrorContains(t, err, "resolver outage")

	_, _, err = provider.FeatureExtractionPooledCodec().Encode(
		EncodeParams[hftypes.FeatureExtractionRequest]{
			Context: context.Background(),
			Request: hftypes.FeatureExtractionRequest{Input: "hello"},
			Model:   deepInfraEmbedModel,
		})
	require.ErrorContains(t, err, "resolver outage")
}
