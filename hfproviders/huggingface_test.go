//go:build !integration

package hfproviders

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestHuggingFaceProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	p := NewHuggingFaceProvider()
	require.Equal(t, "huggingface", p.Name())
	require.Empty(t, p.ProviderSuffix())
}

// endpointCase pairs a display name with an endpoint method so every
// HuggingFaceProvider endpoint can be exercised through one table.
type endpointCase struct {
	name string
	call func(EndpointParams) (Endpoint, error)
	// wantPath computes the expected path from the model.
	wantPath func(model string) string
	// modelRequired reports whether the endpoint rejects an empty model
	// with a configuration error. Only the fixed chat path tolerates one.
	modelRequired bool
}

func endpointCases() []endpointCase {
	modelPath := func(m string) string { return hfModelPrefix + m }

	e := NewHuggingFaceProvider()
	pipeline := func(m string) string { return hfModelPrefix + m + "/pipeline/feature-extraction" }

	// ep builds a model-dependent case for the endpoints whose path is the
	// bare model prefix.
	ep := func(
		name string,
		call func(EndpointParams) (Endpoint, error),
	) endpointCase {
		return endpointCase{name: name, call: call, wantPath: modelPath, modelRequired: true}
	}

	return []endpointCase{
		{
			name:     "ChatEndpoint",
			call:     e.ChatEndpoint,
			wantPath: func(string) string { return "v1/chat/completions" },
		},
		{
			name:          "FeatureExtractionEndpoint",
			call:          e.FeatureExtractionEndpoint,
			wantPath:      pipeline,
			modelRequired: true,
		},
		{
			name:          "FeatureExtractionBatchEndpoint",
			call:          e.FeatureExtractionBatchEndpoint,
			wantPath:      pipeline,
			modelRequired: true,
		},
		ep("ObjectDetectionEndpoint", e.ObjectDetectionEndpoint),
		ep("ImageClassificationEndpoint", e.ImageClassificationEndpoint),
		ep("ImageSegmentationEndpoint", e.ImageSegmentationEndpoint),
		ep("TextClassificationEndpoint", e.TextClassificationEndpoint),
		ep("ZeroShotTextClassificationEndpoint", e.ZeroShotTextClassificationEndpoint),
		ep("TokenClassificationEndpoint", e.TokenClassificationEndpoint),
		ep("QuestionAnsweringEndpoint", e.QuestionAnsweringEndpoint),
		ep("TableQuestionAnsweringEndpoint", e.TableQuestionAnsweringEndpoint),
		ep("TextToImageEndpoint", e.TextToImageEndpoint),
		ep("FillMaskEndpoint", e.FillMaskEndpoint),
		ep("SummarizationEndpoint", e.SummarizationEndpoint),
		ep("TranslationEndpoint", e.TranslationEndpoint),
		ep("SpeechRecognitionEndpoint", e.SpeechRecognitionEndpoint),
	}
}

func TestHuggingFaceProvider_ValidModel(t *testing.T) {
	t.Parallel()

	for _, tc := range endpointCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ep, err := tc.call(EndpointParams{Model: "org/test-model"})
			require.NoError(t, err)
			require.Equal(t, http.MethodPost, ep.Method)
			require.Equal(t, tc.wantPath("org/test-model"), ep.Path)
		})
	}
}

func TestHuggingFaceProvider_EmptyModel(t *testing.T) {
	t.Parallel()

	for _, tc := range endpointCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ep, err := tc.call(EndpointParams{Model: ""})
			if !tc.modelRequired {
				require.NoError(t, err)
				require.Equal(t, tc.wantPath(""), ep.Path)

				return
			}

			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			require.ErrorContains(t, err, errModelIsRequired)
		})
	}
}

// TestHuggingFaceProvider_CodecWiring asserts every codec accessor returns the
// expected built-in codec implementation, pinning the type wiring of the
// provider and its mixins. JSON codecs are wrapped by [hfParamsCodec] so the
// requests' ProviderParams entries merge into the body's top level.
func TestHuggingFaceProvider_CodecWiring(t *testing.T) {
	t.Parallel()

	c := NewHuggingFaceProvider()

	require.IsType(t, hfParamsCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}, c.ChatCodec())
	require.IsType(
		t,
		hfParamsCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		c.ChatStreamCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.ObjectDetectionRequest, []hftypes.ObjectDetection]{},
		c.ObjectDetectionCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionPooledEmbedding,
		]{},
		c.FeatureExtractionPooledBatchCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionPooledEmbedding,
		]{},
		c.FeatureExtractionPooledCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[
			hftypes.FeatureExtractionBatchRequest,
			[]hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		c.FeatureExtractionTokenBatchCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[
			hftypes.FeatureExtractionRequest,
			hftypes.FeatureExtractionTokenEmbeddings,
		]{},
		c.FeatureExtractionTokenCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.FillMaskRequest, []hftypes.FillMaskPrediction]{},
		c.FillMaskCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.ImageClassificationRequest, []hftypes.ImageClassification]{},
		c.ImageClassificationCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.ImageSegmentationRequest, []hftypes.ImageSegmentation]{},
		c.ImageSegmentationCodec(),
	)
	require.IsType(t, hfQuestionAnsweringCodec{}, c.QuestionAnsweringCodec())
	require.IsType(
		t,
		hfParamsCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{},
		c.SummarizationCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.TableQuestionAnsweringRequest, hftypes.TableQuestionAnswer]{},
		c.TableQuestionAnsweringCodec(),
	)
	require.IsType(t, hfTextClassificationCodec{}, c.TextClassificationCodec())
	require.IsType(t, hfTextToImageCodec{}, c.TextToImageCodec())
	require.IsType(
		t,
		hfParamsCodec[hftypes.TokenClassificationRequest, []hftypes.TokenClassification]{},
		c.TokenClassificationCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.TranslationRequest, []hftypes.Translation]{},
		c.TranslationCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.ZeroShotTextClassificationRequest, []hftypes.ZeroShotTextClassification]{},
		c.ZeroShotTextClassificationCodec(),
	)
	require.IsType(t, hfSpeechRecognitionCodec{}, c.SpeechRecognitionCodec())
}

func TestSpeechRecognitionCodec_Encode(t *testing.T) {
	t.Parallel()

	type encodeCase struct {
		name     string
		request  hftypes.SpeechRecognitionRequest
		validate func(*testing.T, []byte, http.Header, error)
	}

	cases := []encodeCase{
		{
			name:    "no params",
			request: hftypes.SpeechRecognitionRequest{Input: "AAAA"},
			validate: func(t *testing.T, body []byte, headers http.Header, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, body)
				require.Len(t, body, 3)
				require.Equal(t, byte(0), body[0])
				require.Equal(t, byte(0), body[1])
				require.Equal(t, byte(0), body[2])
				require.NotEmpty(t, headers.Get("Content-Type"))
			},
		},
		{
			name: "with params",
			request: hftypes.SpeechRecognitionRequest{
				Input: "AAAA",
				Parameters: &hftypes.SpeechRecognitionParameters{
					ReturnTimestamps: new(true),
				},
			},
			validate: func(t *testing.T, body []byte, headers http.Header, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, body)
				require.Equal(t, "application/json", headers.Get("Content-Type"))

				var payload map[string]any
				err = json.Unmarshal(body, &payload)
				require.NoError(t, err)
				inputs, ok := payload["inputs"].(string)
				require.True(t, ok)
				require.Equal(t, "AAAA", inputs)
			},
		},
		{
			name: "provider params only",
			request: hftypes.SpeechRecognitionRequest{
				Input: "AAAA",
				ProviderParams: map[string]any{
					"return_timestamps": true,
				},
			},
			validate: func(t *testing.T, body []byte, headers http.Header, err error) {
				t.Helper()
				require.NoError(t, err)
				require.NotNil(t, body)
				require.Equal(t, "application/json", headers.Get("Content-Type"))
				require.JSONEq(t, `{"inputs":"AAAA","return_timestamps":true}`, string(body))
			},
		},
		{
			name:    "invalid base64",
			request: hftypes.SpeechRecognitionRequest{Input: "!!!invalid"},
			validate: func(t *testing.T, _ []byte, _ http.Header, err error) {
				t.Helper()
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
			},
		},
	}

	codec := hfSpeechRecognitionCodec{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			body, headers, err := codec.Encode(EncodeParams[hftypes.SpeechRecognitionRequest]{
				Request: tc.request,
			})
			tc.validate(t, body, headers, err)
		})
	}
}

func TestSpeechRecognitionCodec_Decode(t *testing.T) {
	t.Parallel()

	codec := hfSpeechRecognitionCodec{}
	resp, err := codec.Decode(DecodeParams{
		Body:    []byte(`{"text":"hello world"}`),
		Headers: http.Header{"Content-Type": {"application/json"}},
	})
	require.NoError(t, err)
	require.Equal(t, "hello world", resp.Text)
	require.Nil(t, resp.Chunks)
}

// TestHuggingFaceProvider_TaskCoverage asserts the Hugging Face provider does
// not advertise tasks it does not serve. Text-to-video and image-to-image are
// served through partner providers only, e.g. [WaveSpeedProvider].
func TestHuggingFaceProvider_TaskCoverage(t *testing.T) {
	t.Parallel()

	p := NewHuggingFaceProvider()

	_, err := AsProvider[ImageToImageProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"huggingface does not implement hfproviders.ImageToImageProvider",
	)

	_, err = AsProvider[TextToVideoProvider](p)
	require.Error(t, err)
	require.ErrorContains(
		t,
		err,
		"huggingface does not implement hfproviders.TextToVideoProvider",
	)
}

// TestHuggingFaceProvider_ImplementsEveryTaskProvider pins the aggregate's
// method promotion. Because every per-task method name is unique to its mixin,
// these assertions also confirm each mixin carries the methods its interface
// requires.
func TestHuggingFaceProvider_ImplementsEveryTaskProvider(t *testing.T) {
	t.Parallel()

	var (
		_ ChatProvider                         = HuggingFaceProvider{}
		_ ChatStreamProvider                   = HuggingFaceProvider{}
		_ FeatureExtractionPooledBatchProvider = HuggingFaceProvider{}
		_ FeatureExtractionPooledProvider      = HuggingFaceProvider{}
		_ FeatureExtractionTokenProvider       = HuggingFaceProvider{}
		_ FeatureExtractionTokenBatchProvider  = HuggingFaceProvider{}
		_ FillMaskProvider                     = HuggingFaceProvider{}
		_ ImageClassificationProvider          = HuggingFaceProvider{}
		_ ObjectDetectionProvider              = HuggingFaceProvider{}
		_ ImageSegmentationProvider            = HuggingFaceProvider{}
		_ QuestionAnsweringProvider            = HuggingFaceProvider{}
		_ SummarizationProvider                = HuggingFaceProvider{}
		_ TableQuestionAnsweringProvider       = HuggingFaceProvider{}
		_ TextClassificationProvider           = HuggingFaceProvider{}
		_ TextToImageProvider                  = HuggingFaceProvider{}
		_ TokenClassificationProvider          = HuggingFaceProvider{}
		_ TranslationProvider                  = HuggingFaceProvider{}
		_ ZeroShotTextClassificationProvider   = HuggingFaceProvider{}
		_ SpeechRecognitionProvider            = HuggingFaceProvider{}
	)
}

// chatEncodeTestModel is the model the chat encode tests' expected bodies
// pin.
const chatEncodeTestModel = "mistralai/Mistral-7B-Instruct-v0.3"

// newChatEncodeTestRequest builds a minimal valid chat request for codec
// encode tests.
func newChatEncodeTestRequest() hftypes.ChatRequest {
	return hftypes.ChatRequest{
		Model: new(chatEncodeTestModel),
		Messages: []hftypes.ChatMessage{{
			Role:    "user",
			Content: hftypes.ChatMessageContent{Text: new("hello")},
		}},
	}
}

// TestHuggingFaceChatCodecs_EncodeProviderParams exercises the chat codecs'
// ProviderParams merge: entries land at the body's top level, entries override
// typed fields, the resolved "model" is protected, bodies without
// ProviderParams stay byte-identical, and numeric fidelity survives the
// body's map round-trip.
func TestHuggingFaceChatCodecs_EncodeProviderParams(t *testing.T) {
	t.Parallel()

	p := NewHuggingFaceProvider()

	t.Run("merges entries into the body and protects the model", func(t *testing.T) {
		t.Parallel()

		request := newChatEncodeTestRequest()
		request.Seed = new(int64(42))
		request.ProviderParams = map[string]any{
			"frequency_penalty": 0.5,
			"model":             "attacker-model",
		}

		body, headers, err := p.ChatCodec().
			Encode(EncodeParams[hftypes.ChatRequest]{Request: request})
		require.NoError(t, err)
		require.Equal(t, http.Header{
			headerAccept:   {mimeTypeJSON},
			"Content-Type": {mimeTypeJSON},
		}, headers)
		require.JSONEq(t, fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":"hello"}],"seed":42,"frequency_penalty":0.5}`,
			chatEncodeTestModel,
		), string(body))
	})

	t.Run("protects the model on the stream codec", func(t *testing.T) {
		t.Parallel()

		request := newChatEncodeTestRequest()
		request.ProviderParams = map[string]any{"model": "attacker-model"}

		body, _, err := p.ChatStreamCodec().
			Encode(EncodeParams[hftypes.ChatRequest]{Request: request})
		require.NoError(t, err)
		require.JSONEq(t, fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`,
			chatEncodeTestModel,
		), string(body))
	})

	t.Run("no-op when ProviderParams is nil", func(t *testing.T) {
		t.Parallel()

		chatRequest := newChatEncodeTestRequest()
		body, _, err := p.ChatCodec().
			Encode(EncodeParams[hftypes.ChatRequest]{Request: chatRequest})
		require.NoError(t, err)
		direct, err := json.Marshal(chatRequest)
		require.NoError(t, err)
		require.Equal(t, string(direct), string(body))

		summarizationRequest := hftypes.SummarizationRequest{Input: "text"}
		body, _, err = p.SummarizationCodec().
			Encode(EncodeParams[hftypes.SummarizationRequest]{Request: summarizationRequest})
		require.NoError(t, err)
		direct, err = json.Marshal(summarizationRequest)
		require.NoError(t, err)
		require.Equal(t, string(direct), string(body))
	})

	t.Run("round-trips numbers exactly", func(t *testing.T) {
		t.Parallel()

		request := newChatEncodeTestRequest()
		request.Seed = new(int64(9007199254740993))
		request.ProviderParams = map[string]any{"temperature": 1}

		body, _, err := p.ChatCodec().Encode(EncodeParams[hftypes.ChatRequest]{Request: request})
		require.NoError(t, err)
		require.Contains(t, string(body), "9007199254740993")
		require.Contains(t, string(body), `"temperature":1`)
	})

	t.Run("rejects unmarshalable ProviderParams entries", func(t *testing.T) {
		t.Parallel()

		request := newChatEncodeTestRequest()
		request.ProviderParams = map[string]any{"bad": func() {}}

		_, _, err := p.ChatCodec().Encode(EncodeParams[hftypes.ChatRequest]{Request: request})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}

// TestHuggingFaceSummarizationCodec_EncodeProviderParams exercises the plain
// pipeline merge path through one typed task: top-level merging, typed-field
// overrides, the protected "inputs" key, and the wholesale-"parameters"
// replacement nuance.
func TestHuggingFaceSummarizationCodec_EncodeProviderParams(t *testing.T) {
	t.Parallel()

	codec := hfPipelineParamsCodec(
		JSONCodec[hftypes.SummarizationRequest, []hftypes.Summarization]{},
		summarizationRequestProviderParams,
	)

	t.Run("merges entries into the body's top level", func(t *testing.T) {
		t.Parallel()

		request := hftypes.SummarizationRequest{
			Input:          "text",
			ProviderParams: map[string]any{"max_new_tokens": 128},
		}

		body, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"text","max_new_tokens":128}`, string(body))
	})

	t.Run("leaves nested typed parameters untouched", func(t *testing.T) {
		t.Parallel()

		// Top-level entries cannot reach into the nested "parameters" object:
		// a "truncation" entry rides at the body's top level alongside the
		// typed nested parameters. Overriding a nested typed value requires a
		// wholesale "parameters" replacement (see the next subtest).
		request := hftypes.SummarizationRequest{
			Input: "text",
			Parameters: &hftypes.SummarizationParameters{
				Truncation: new(hftypes.SummarizationTruncationLongestFirst),
			},
			ProviderParams: map[string]any{
				"truncation": hftypes.SummarizationTruncationDoNotTruncate,
			},
		}

		body, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"inputs":"text","parameters":{"truncation":"longest_first"},"truncation":"do_not_truncate"}`,
			string(body),
		)
	})

	t.Run("protects the inputs key", func(t *testing.T) {
		t.Parallel()

		request := hftypes.SummarizationRequest{
			Input:          "text",
			ProviderParams: map[string]any{"inputs": "evil"},
		}

		body, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"text"}`, string(body))
	})

	t.Run("replaces the typed parameters object wholesale", func(t *testing.T) {
		t.Parallel()

		request := hftypes.SummarizationRequest{
			Input: "text",
			Parameters: &hftypes.SummarizationParameters{
				Truncation: new(hftypes.SummarizationTruncationLongestFirst),
			},
			ProviderParams: map[string]any{
				"parameters": map[string]any{"custom": true},
			},
		}

		body, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"text","parameters":{"custom":true}}`, string(body))
	})

	t.Run("rejects unmarshalable ProviderParams entries", func(t *testing.T) {
		t.Parallel()

		request := hftypes.SummarizationRequest{
			Input:          "text",
			ProviderParams: map[string]any{"bad": func() {}},
		}

		_, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("does not mutate the caller's ProviderParams map", func(t *testing.T) {
		t.Parallel()

		request := hftypes.SummarizationRequest{
			Input:          "text",
			ProviderParams: map[string]any{"max_new_tokens": 128},
		}
		before := maps.Clone(request.ProviderParams)

		_, _, err := codec.Encode(EncodeParams[hftypes.SummarizationRequest]{Request: request})
		require.NoError(t, err)
		require.True(t, maps.Equal(before, request.ProviderParams))
	})
}

// TestHuggingFacePipelineCodecs_EncodeProviderParams exercises the remaining
// distinct HF codec encode paths: the feature-extraction wrapper, the
// response-shape-flexible delegation codecs (question answering and text
// classification), and the text-to-image raw-response codec.
func TestHuggingFacePipelineCodecs_EncodeProviderParams(t *testing.T) {
	t.Parallel()

	t.Run("feature extraction", func(t *testing.T) {
		t.Parallel()

		codec := NewHuggingFaceProvider().FeatureExtractionPooledCodec()
		request := hftypes.FeatureExtractionRequest{
			Input:          "text",
			ProviderParams: map[string]any{"normalize": true},
		}

		body, _, err := codec.Encode(
			EncodeParams[hftypes.FeatureExtractionRequest]{Request: request},
		)
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"text","normalize":true}`, string(body))
	})

	t.Run("question answering", func(t *testing.T) {
		t.Parallel()

		codec := hfQuestionAnsweringCodec{}
		request := hftypes.QuestionAnsweringRequest{
			Input: hftypes.QuestionAnsweringInput{Question: "who?", Context: "ctx"},
			ProviderParams: map[string]any{
				"top_k":  3,
				"inputs": "evil",
			},
		}

		body, _, err := codec.Encode(
			EncodeParams[hftypes.QuestionAnsweringRequest]{Request: request},
		)
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"inputs":{"context":"ctx","question":"who?"},"top_k":3}`,
			string(body),
		)
	})

	t.Run("text classification", func(t *testing.T) {
		t.Parallel()

		codec := hfTextClassificationCodec{}
		request := hftypes.TextClassificationRequest{
			Input:          "text",
			ProviderParams: map[string]any{"top_k": 2},
		}

		body, _, err := codec.Encode(
			EncodeParams[hftypes.TextClassificationRequest]{Request: request},
		)
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"text","top_k":2}`, string(body))
	})

	t.Run("text to image", func(t *testing.T) {
		t.Parallel()

		codec := NewHuggingFaceProvider().TextToImageCodec()
		request := hftypes.TextToImageRequest{
			Input:          "a corgi",
			ProviderParams: map[string]any{"negative_prompt": "blurry"},
		}

		body, headers, err := codec.Encode(
			EncodeParams[hftypes.TextToImageRequest]{Request: request},
		)
		require.NoError(t, err)
		require.JSONEq(t, `{"inputs":"a corgi","negative_prompt":"blurry"}`, string(body))
		require.Equal(t, http.Header{
			headerAccept:   {mimeTypeAny},
			"Content-Type": {mimeTypeJSON},
		}, headers)
	})
}
