//go:build !integration

package hfproviders

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const togetherASRModel = "nvidia/nemotron-3.5-asr-streaming-0.6b"

// newTogetherProvider builds a TogetherProvider backed by a static resolver
// covering the test model.
func newTogetherProvider(t *testing.T) TogetherProvider {
	t.Helper()

	provider, err := NewTogetherProvider(NewStaticModelResolver(map[StaticMapping]ModelMapping{
		{
			Provider: togetherProviderName,
			Task:     TaskAutomaticSpeechRecognition,
			Model:    togetherASRModel,
		}: {ProviderModelID: "together-ai/nemotron-asr"},
	}))
	require.NoError(t, err)

	return provider
}

// togetherChatRequest builds the minimal valid chat request; the dispatch
// layer always fills Model before the codec sees it.
func togetherChatRequest() hftypes.ChatRequest {
	text := "hi"

	return hftypes.ChatRequest{
		Model: new("deepseek-ai/DeepSeek-V4.1-Flash:together"),
		Messages: []hftypes.ChatMessage{
			{
				Role:    "user",
				Content: hftypes.ChatMessageContent{Text: &text},
			},
		},
	}
}

func TestTogetherProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	provider := newTogetherProvider(t)

	require.Equal(t, "together", provider.Name())
	require.Equal(t, "together", provider.ProviderSuffix())
}

func TestNewTogetherProvider_NilResolver(t *testing.T) {
	t.Parallel()

	provider, err := NewTogetherProvider(nil)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Equal(t, TogetherProvider{}, provider)
}

// TestTogetherProvider_ZeroValueIsSafe verifies the zero-value provider's
// speech-recognition endpoint and standalone codec fail with configuration
// errors instead of dereferencing a nil resolver, while chat keeps working.
func TestTogetherProvider_ZeroValueIsSafe(t *testing.T) {
	t.Parallel()

	var provider TogetherProvider

	_, err := provider.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "any-model",
	})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	_, _, err = provider.SpeechRecognitionCodec().Encode(
		EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Model:   "any-model",
		})
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)

	// Chat needs no resolver: the zero value still routes.
	endpoint, err := provider.ChatEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "model:together",
	})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
}

// TestTogetherProvider_Contract pins the task interfaces TogetherProvider
// serves and the ones it must reject.
func TestTogetherProvider_Contract(t *testing.T) {
	t.Parallel()

	provider := newTogetherProvider(t)

	for _, served := range []func(Provider) error{
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
	} {
		require.NoError(t, served(provider))
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
			typeName: "hfproviders.TextToImageProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextToImageProvider](p)

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
			typeName: "hfproviders.TokenClassificationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TokenClassificationProvider](p)

				return err
			},
		},
	} {
		err := missing.cast(provider)
		require.Error(t, err, "must not implement %s", missing.typeName)
		require.ErrorContains(t, err, "together does not implement "+missing.typeName)
	}
}

func TestTogetherProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	provider := newTogetherProvider(t)

	chat, err := AsProvider[ChatProvider](provider)
	require.NoError(t, err)

	// Chat routes through the fixed OpenAI-compatible path, model-in-body:
	// an empty Model is legal and no mapping is attached.
	endpoint, err := chat.ChatEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   "deepseek-ai/DeepSeek-V4.1-Flash:together",
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
	require.Nil(t, endpoint.Mapping)

	endpoint, err = chat.ChatEndpoint(EndpointParams{Context: context.Background()})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)

	// Codec wiring: the Together-specific codecs, pinning that the provider
	// rewrites the chat request wire format rather than reusing the plain
	// JSON codec.
	require.IsType(t, togetherChatCodec{}, chat.ChatCodec())

	stream, err := AsProvider[ChatStreamProvider](provider)
	require.NoError(t, err)
	require.IsType(t, togetherChatStreamCodec{}, stream.ChatStreamCodec())
}

// TestTogetherMixin_SubsetProvider verifies the speech-recognition mixin can
// be embedded in a custom provider that only implements the identity methods.
func TestTogetherMixin_SubsetProvider(t *testing.T) {
	t.Parallel()

	provider := togetherSubsetProvider{
		TogetherSpeechRecognitionMixin{
			Resolver: NewStaticModelResolver(map[StaticMapping]ModelMapping{
				{
					Provider: togetherProviderName,
					Task:     TaskAutomaticSpeechRecognition,
					Model:    togetherASRModel,
				}: {ProviderModelID: "together-ai/nemotron-asr"},
			}),
		},
	}

	speech, err := AsProvider[SpeechRecognitionProvider](provider)
	require.NoError(t, err)

	endpoint, err := speech.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   togetherASRModel,
	})
	require.NoError(t, err)
	require.Equal(t, "together/v1/audio/transcriptions", endpoint.Path)

	_, err = AsProvider[ChatProvider](provider)
	require.Error(t, err)
	require.ErrorContains(t, err, "together-subset does not implement hfproviders.ChatProvider")
}

// togetherSubsetProvider serves only speech recognition by embedding the
// mixin and supplying the provider identity.
type togetherSubsetProvider struct {
	TogetherSpeechRecognitionMixin
}

// Compile-time assertion that the subset provider satisfies the interface.
var _ SpeechRecognitionProvider = (*togetherSubsetProvider)(nil)

func (togetherSubsetProvider) Name() string           { return "together-subset" }
func (togetherSubsetProvider) ProviderSuffix() string { return "" }

func TestTogetherProvider_SpeechRecognitionEndpoint(t *testing.T) {
	t.Parallel()

	resolver := &deepInfraRecordingResolver{
		mapping: ModelMapping{
			ProviderModelID: "together-ai/nemotron-asr",
			Status:          "live",
		},
	}
	provider, err := NewTogetherProvider(resolver)
	require.NoError(t, err)

	speech, err := AsProvider[SpeechRecognitionProvider](provider)
	require.NoError(t, err)

	endpoint, err := speech.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   togetherASRModel,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "together/v1/audio/transcriptions", endpoint.Path)
	require.Equal(t, "together-ai/nemotron-asr", endpoint.Mapping.ProviderModelID)

	require.Equal(t, ResolveModelParams{
		Provider: togetherProviderName,
		Task:     TaskAutomaticSpeechRecognition,
		Model:    togetherASRModel,
	}, resolver.got)
}

func TestTogetherSpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	provider := newTogetherProvider(t)
	codec := provider.SpeechRecognitionCodec()

	t.Run("multipart body carries audio, parameters, and mapped model", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
				ProviderParams: map[string]any{"language": "en"},
			},
			Model: togetherASRModel,
		})
		require.NoError(t, err)

		contentType := headers.Get("Content-Type")
		parts := deepInfraParseMultipart(t, contentType, body)
		require.Len(t, parts, 4)

		file := deepInfraFindPart(t, parts, "file")
		require.Equal(t, "audio.wav", file.filename)
		require.Equal(t, []byte(deepInfraWAVAudio), file.value)
		require.Equal(t, "true", string(deepInfraFindPart(t, parts, "return_timestamps").value))
		require.Equal(t, "en", string(deepInfraFindPart(t, parts, "language").value))
		// The mapped provider identifier rides on the model field.
		require.Equal(
			t,
			"together-ai/nemotron-asr",
			string(deepInfraFindPart(t, parts, "model").value),
		)
	})

	t.Run("endpoint-resolved mapping wins over the resolver", func(t *testing.T) {
		t.Parallel()

		resolver := &deepInfraRecordingResolver{mapping: ModelMapping{ProviderModelID: "unused"}}
		togetherCodec := togetherSpeechRecognitionCodec{resolver: resolver}

		body, headers, err := togetherCodec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
			},
			Model:   togetherASRModel,
			Mapping: &ModelMapping{ProviderModelID: "supplied-model"},
		})
		require.NoError(t, err)
		require.Empty(t, resolver.got.Model, "supplied mapping must skip the resolver")

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.Equal(t, "supplied-model", string(deepInfraFindPart(t, parts, "model").value))
	})

	t.Run("provider parameters cannot override the model", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input:          base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
				ProviderParams: map[string]any{"model": "attacker-model"},
			},
			Model: togetherASRModel,
		})
		require.NoError(t, err)

		parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
		require.Equal(
			t,
			"together-ai/nemotron-asr",
			string(deepInfraFindPart(t, parts, "model").value),
		)
	})

	t.Run("invalid base64 input fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{Input: "not base64!!"},
			Model:   togetherASRModel,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode base64 input")
	})
}

func TestTogetherSpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	jsonHeaders := func() http.Header {
		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		return headers
	}

	resp, err := TogetherSpeechRecognitionMixin{}.SpeechRecognitionCodec().Decode(DecodeParams{
		Context: context.Background(),
		Body: []byte(
			`{"text":"ask not what your country can do for you","segments":[` +
				`{"id":0,"start":0.0,"end":2.5,"text":"ask not"}],"language":"en"}`,
		),
		Headers: jsonHeaders(),
	})
	require.NoError(t, err)
	require.Equal(t, "ask not what your country can do for you", resp.Text)
	require.Len(t, resp.Chunks, 1)
	require.Equal(t, []float64{0.0, 2.5}, resp.Chunks[0].Timestamp)
	require.Equal(t, map[string]any{"language": "en"}, resp.ProviderMetadata)
}

// TestTogetherSpeechRecognitionCodec_ResolverErrorsPropagate pins that a
// resolver error (not just a nil resolver) surfaces from the endpoint and
// from the standalone codec.
func TestTogetherSpeechRecognitionCodec_ResolverErrorsPropagate(t *testing.T) {
	t.Parallel()

	provider, err := NewTogetherProvider(deepInfraFailingResolver{})
	require.NoError(t, err)

	_, err = provider.SpeechRecognitionEndpoint(EndpointParams{
		Context: context.Background(),
		Model:   togetherASRModel,
	})
	require.ErrorContains(t, err, "resolver outage")

	_, _, err = provider.SpeechRecognitionCodec().Encode(
		EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte(deepInfraWAVAudio)),
			},
			Model: togetherASRModel,
		})
	require.ErrorContains(t, err, "resolver outage")
}

// TestTogetherChatCodec_ResponseFormatRewrite pins Together's chat request
// wire format: json_schema response formats are flattened, everything else
// passes through exactly as the default provider's JSON codec emits it.
func TestTogetherChatCodec_ResponseFormatRewrite(t *testing.T) {
	t.Parallel()

	codec := togetherChatCodec{}

	encode := func(t *testing.T, req hftypes.ChatRequest) map[string]json.RawMessage {
		t.Helper()

		body, _, err := codec.Encode(EncodeParams[hftypes.ChatRequest]{
			Context: context.Background(),
			Request: req,
		})
		require.NoError(t, err)

		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(body, &fields))

		return fields
	}

	t.Run("json_schema envelope is flattened", func(t *testing.T) {
		t.Parallel()

		schema := json.RawMessage(
			`{"type":"object","properties":{"answer":{"type":"string"}}}`,
		)
		req := togetherChatRequest()
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{
				Name:   "answer",
				Strict: new(true),
				Schema: schema,
			},
		}

		fields := encode(t, req)

		require.JSONEq(t,
			`{"type":"json_schema","schema":`+string(schema)+`}`,
			string(fields["response_format"]),
		)

		// The flattened shape carries neither the envelope nor name/strict.
		var format struct {
			Type       string          `json:"type"`
			Schema     json.RawMessage `json:"schema"`
			JSONSchema json.RawMessage `json:"json_schema"`
		}
		require.NoError(t, json.Unmarshal(fields["response_format"], &format))
		require.Equal(t, "json_schema", format.Type)
		require.NotNil(t, format.Schema)
		require.Nil(t, format.JSONSchema)
	})

	t.Run("json_schema without a schema keeps the envelope", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type:       hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "answer"},
		}

		fields := encode(t, req)

		require.NotContains(t, string(fields["response_format"]), `"schema":`)
		require.Contains(t, string(fields["response_format"]), `"json_schema"`)
	})

	t.Run("text and json_object pass through", func(t *testing.T) {
		t.Parallel()

		for _, formatType := range []hftypes.ResponseFormatType{
			hftypes.ResponseFormatTypeText,
			hftypes.ResponseFormatTypeJSONObject,
		} {
			req := togetherChatRequest()
			req.ResponseFormat = &hftypes.ChatResponseFormat{Type: formatType}

			fields := encode(t, req)
			require.JSONEq(t,
				`{"type":"`+string(formatType)+`"}`,
				string(fields["response_format"]),
			)
		}
	})

	t.Run("provider-specific format types pass through", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatType("audio"),
		}

		fields := encode(t, req)
		require.JSONEq(t, `{"type":"audio"}`, string(fields["response_format"]))
	})

	t.Run("absent response format yields no key", func(t *testing.T) {
		t.Parallel()

		fields := encode(t, togetherChatRequest())
		_, present := fields["response_format"]
		require.False(t, present)
	})

	t.Run("messages and other fields are unchanged", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		req.MaxTokens = new(512)
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{
				Name:   "a",
				Schema: json.RawMessage(`{"type":"object"}`),
			},
		}

		fields := encode(t, req)

		require.JSONEq(t, string(mustMarshal(req.Messages)), string(fields["messages"]))
		require.JSONEq(t, "512", string(fields["max_tokens"]))
	})

	t.Run("vision-language chunks pass through", func(t *testing.T) {
		t.Parallel()

		imgURL := "https://example.com/img.png"
		text := "hi"
		req := hftypes.ChatRequest{
			Model: new("deepseek-ai/DeepSeek-V4.1-Flash:together"),
			Messages: []hftypes.ChatMessage{{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Chunks: []hftypes.ChatMessageChunk{
						{Type: hftypes.MessageChunkTypeText, Text: &text},
						{
							Type:     hftypes.MessageChunkTypeImageURL,
							ImageURL: &hftypes.ChatImageURL{URL: imgURL},
						},
					},
				},
			}},
		}

		fields := encode(t, req)
		require.Contains(t, string(fields["messages"]), imgURL)
	})

	t.Run("invalid request fails before the rewrite", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		// json_schema requires the envelope, so this fails MarshalJSON.
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
		}

		_, _, err := codec.Encode(EncodeParams[hftypes.ChatRequest]{
			Context: context.Background(),
			Request: req,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "json_schema requires json_schema field")
	})
}

// TestTogetherChatCodec_UnaryDecode pins that responses decode exactly like
// the default provider's JSON codec, including Together's occasional
// omission of system_fingerprint.
func TestTogetherChatCodec_UnaryDecode(t *testing.T) {
	t.Parallel()

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")

	body := []byte(
		`{"id":"gen-1","object":"chat.completion","created":1700000000,` +
			`"model":"deepseek-ai/DeepSeek-V4.1-Flash","choices":[` +
			`{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	)

	codec := togetherChatCodec{}
	resp, err := codec.Decode(DecodeParams{
		Context: context.Background(),
		Body:    body,
		Headers: headers,
	})
	require.NoError(t, err)
	require.Equal(t, "gen-1", resp.ID)
	require.Equal(t, "hello", *resp.Choices[0].Message.Content)
	require.Empty(t, resp.SystemFingerprint)

	// The rewritten request must still round-trip through the shared types:
	// decode parity with JSONCodec for the same body.
	reference, err := (JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}).Decode(
		DecodeParams{Context: context.Background(), Body: body, Headers: headers},
	)
	require.NoError(t, err)
	require.Equal(t, reference, resp)
}

// TestTogetherChatStreamCodec pins that the streaming codec applies the same
// request rewrite and decodes stream chunks like the default provider.
func TestTogetherChatStreamCodec(t *testing.T) {
	t.Parallel()

	codec := togetherChatStreamCodec{}

	req := togetherChatRequest()
	req.Stream = new(true)
	req.ResponseFormat = &hftypes.ChatResponseFormat{
		Type: hftypes.ResponseFormatTypeJSONSchema,
		JSONSchema: &hftypes.ChatJSONSchemaConfig{
			Name:   "a",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	}

	body, headers, err := codec.Encode(EncodeParams[hftypes.ChatRequest]{
		Context: context.Background(),
		Request: req,
	})
	require.NoError(t, err)
	require.Equal(t, "application/json", headers.Get("Content-Type"))

	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &fields))
	require.JSONEq(t,
		`{"type":"json_schema","schema":{"type":"object"}}`,
		string(fields["response_format"]),
	)
	require.JSONEq(t, "true", string(fields["stream"]))

	headers = http.Header{}
	headers.Set("Content-Type", "application/json")

	chunk, err := codec.Decode(DecodeParams{
		Context: context.Background(),
		Body: []byte(
			`{"id":"gen-2","object":"chat.completion.chunk","created":1700000000,` +
				`"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
		),
		Headers: headers,
	})
	require.NoError(t, err)
	require.Equal(t, "gen-2", chunk.ID)
	require.Equal(t, "hi", *chunk.Choices[0].Delta.Content)

	t.Run("invalid request fails before the rewrite", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		// json_schema requires the envelope, so this fails MarshalJSON.
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
		}

		_, _, err := codec.Encode(EncodeParams[hftypes.ChatRequest]{
			Context: context.Background(),
			Request: req,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "json_schema requires json_schema field")
	})
}

// TestTogetherSpeechCodec_MultipartSniffing pins that Together's ASR codec
// labels a FLAC payload correctly through the shared sniffer.
func TestTogetherSpeechCodec_MultipartSniffing(t *testing.T) {
	t.Parallel()

	provider := newTogetherProvider(t)

	body, headers, err := provider.SpeechRecognitionCodec().Encode(
		EncodeParams[hftypes.SpeechRecognitionRequest]{
			Context: context.Background(),
			Request: hftypes.SpeechRecognitionRequest{
				Input: base64.StdEncoding.EncodeToString([]byte("fLaC\x00\x00\x00\x22")),
			},
			Model: togetherASRModel,
		})
	require.NoError(t, err)

	parts := deepInfraParseMultipart(t, headers.Get("Content-Type"), body)
	file := deepInfraFindPart(t, parts, "file")
	require.Equal(t, "audio.flac", file.filename)
	require.Equal(t, "audio/flac", file.contentType)
}

// TestTogetherFlattenResponseFormat_Errors exercises the rewrite helper's
// defensive error branches directly. The full codec Encode path cannot
// produce such inputs — JSONCodec.Encode marshals (and thereby validates)
// the request first — so the tests craft the arguments by hand, mirroring
// the shared-helper tests.
func TestTogetherFlattenResponseFormat_Errors(t *testing.T) {
	t.Parallel()

	format := &hftypes.ChatResponseFormat{
		Type: hftypes.ResponseFormatTypeJSONSchema,
		JSONSchema: &hftypes.ChatJSONSchemaConfig{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	}

	t.Run("undecodable body fails", func(t *testing.T) {
		t.Parallel()

		_, err := togetherFlattenResponseFormat(format, []byte("not json"))
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "failed to decode request body")
	})

	t.Run("invalid schema raw fails to marshal", func(t *testing.T) {
		t.Parallel()

		bad := &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{
				Name:   "answer",
				Schema: json.RawMessage("{invalid"),
			},
		}

		_, err := togetherFlattenResponseFormat(bad, []byte(`{"model":"m"}`))
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, errFailedToMarshalRequestBody)
	})
}

// mustMarshal panics on failure; test-only helper for comparison targets.
func mustMarshal(value any) []byte {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}

	return encoded
}
