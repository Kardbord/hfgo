//go:build !integration

package hfproviders

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// taskCast pairs a task-interface name with the AsProvider cast that must
// fail for a provider that does not serve the task.
type taskCast struct {
	typeName string
	cast     func(Provider) error
}

// requireServedTaskInterfaces asserts that provider casts to every listed
// task interface without error.
func requireServedTaskInterfaces(t *testing.T, provider Provider, served []func(Provider) error) {
	t.Helper()

	for _, serve := range served {
		require.NoError(t, serve(provider))
	}
}

// requireMissingTaskInterfaces asserts that every listed cast fails with the
// canonical "<slug> does not implement <interface>" configuration error.
func requireMissingTaskInterfaces(
	t *testing.T,
	provider Provider,
	slug string,
	missing []taskCast,
) {
	t.Helper()

	for _, m := range missing {
		err := m.cast(provider)
		require.Error(t, err, "must not implement %s", m.typeName)
		require.ErrorContains(t, err, slug+" does not implement "+m.typeName)
	}
}

// requireOpenAIChatPartnerEndpoint pins the chat half shared by partner
// providers that reuse the default provider's OpenAI-compatible chat endpoint
// and codecs: the endpoint is model-independent, attaches no mapping, and the
// codecs are the built-in JSON chat codecs wrapped with the ProviderParams
// merge (entries land at the body's top level, the resolved "model" is
// protected).
func requireOpenAIChatPartnerEndpoint(t *testing.T, provider Provider, slug string) {
	t.Helper()

	chat, err := AsProvider[ChatProvider](provider)
	require.NoError(t, err)
	stream, err := AsProvider[ChatStreamProvider](provider)
	require.NoError(t, err)

	// Chat routes through the fixed OpenAI-compatible path, model-in-body:
	// an empty Model is legal and no mapping is attached.
	endpoint, err := chat.ChatEndpoint(context.Background(), EndpointParams{
		Model: "zai-org/GLM-5.3:" + slug,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, "v1/chat/completions", endpoint.Path)
	require.Nil(t, endpoint.Mapping)

	endpoint, err = chat.ChatEndpoint(context.Background(), EndpointParams{})
	require.NoError(t, err)
	require.Equal(t, "v1/chat/completions", endpoint.Path)

	// Codec wiring: the built-in JSON chat codecs wrapped with the
	// ProviderParams merge, pinning that chat reuses the default
	// OpenAI-compatible wire format.
	require.IsType(
		t,
		hfParamsCodec[hftypes.ChatRequest, hftypes.ChatResponse]{},
		chat.ChatCodec(),
	)
	require.IsType(
		t,
		hfParamsCodec[hftypes.ChatRequest, hftypes.ChatStreamResponse]{},
		stream.ChatStreamCodec(),
	)

	// ProviderParams entries merge into the body's top level and cannot
	// override the resolved model.
	model := "zai-org/GLM-5.3:" + slug
	request := hftypes.ChatRequest{
		Model: new(model),
		Messages: []hftypes.ChatMessage{{
			Role:    "user",
			Content: hftypes.ChatMessageContent{Text: new("hello")},
		}},
		ProviderParams: map[string]any{
			"temperature": 0.5,
			"model":       "attacker-model",
		},
	}
	body, _, err := chat.ChatCodec().
		Encode(context.Background(), EncodeParams[hftypes.ChatRequest]{Request: request})
	require.NoError(t, err)
	require.JSONEq(t, fmt.Sprintf(
		`{"model":%q,"messages":[{"role":"user","content":"hello"}],"temperature":0.5}`,
		model,
	), string(body))
}

// requireOpenAIEmbeddingsEndpoints pins the embeddings half shared by partner
// providers speaking the OpenAI embeddings wire format: single and batch
// requests resolve the same router-proxied path with the provider-side model
// identifier attached as the endpoint mapping.
func requireOpenAIEmbeddingsEndpoints(
	t *testing.T,
	provider Provider,
	model, path, providerModelID string,
) {
	t.Helper()

	pooled, err := AsProvider[FeatureExtractionPooledProvider](provider)
	require.NoError(t, err)
	batch, err := AsProvider[FeatureExtractionPooledBatchProvider](provider)
	require.NoError(t, err)

	endpoint, err := pooled.FeatureExtractionEndpoint(context.Background(), EndpointParams{
		Model: model,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, path, endpoint.Path)
	require.Equal(t, providerModelID, endpoint.Mapping.ProviderModelID)

	endpoint, err = batch.FeatureExtractionBatchEndpoint(context.Background(), EndpointParams{
		Model: model,
	})
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, endpoint.Method)
	require.Equal(t, path, endpoint.Path)
	require.Equal(t, providerModelID, endpoint.Mapping.ProviderModelID)
}

// TestOpenAIEmbeddingsCodecs_Decode pins the decode behavior every provider
// codec family speaking the OpenAI embeddings wire format shares: single
// codecs return the first data entry's embedding, batch codecs return one
// embedding per entry in order, and malformed responses fail with
// serialization errors. Add a case per new provider adopting the format.
func TestOpenAIEmbeddingsCodecs_Decode(t *testing.T) {
	t.Parallel()

	type singleCodecCase struct {
		name  string
		codec FeatureExtractionPooledCodec
	}

	singles := []singleCodecCase{
		{name: "deepinfra", codec: deepinfraFeatureExtractionCodec{}},
		{name: "scaleway", codec: scalewayFeatureExtractionCodec{}},
	}

	type batchCodecCase struct {
		name  string
		codec FeatureExtractionPooledBatchCodec
	}

	batches := []batchCodecCase{
		{name: "deepinfra", codec: deepinfraFeatureExtractionBatchCodec{}},
		{name: "scaleway", codec: scalewayFeatureExtractionBatchCodec{}},
	}

	t.Run("single returns the first embedding", func(t *testing.T) {
		t.Parallel()

		for _, tc := range singles {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				embedding, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body: []byte(
						`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[1,2,3]},{"object":"embedding","index":1,"embedding":[4,5,6]}],"model":"m"}`,
					),
					Headers: jsonHeaders(ctJSON),
				})
				require.NoError(t, err)
				require.Equal(t, hftypes.FeatureExtractionPooledEmbedding{1, 2, 3}, embedding)
			})
		}
	})

	t.Run("batch returns every embedding in order", func(t *testing.T) {
		t.Parallel()

		for _, tc := range batches {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				embeddings, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body:    []byte(`{"data":[{"embedding":[1,2]},{"embedding":[3,4]}]}`),
					Headers: jsonHeaders(ctJSON),
				})
				require.NoError(t, err)
				require.Equal(
					t,
					[]hftypes.FeatureExtractionPooledEmbedding{{1, 2}, {3, 4}},
					embeddings,
				)
			})
		}
	})

	t.Run("empty data fails", func(t *testing.T) {
		t.Parallel()

		for _, tc := range singles {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				_, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body:    []byte(`{"data":[]}`),
					Headers: jsonHeaders(ctJSON),
				})
				require.ErrorContains(t, err, "contains no embeddings")
			})
		}
	})

	t.Run("missing embedding fails", func(t *testing.T) {
		t.Parallel()

		for _, tc := range batches {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				_, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body:    []byte(`{"data":[{"embedding":[1]},{"index":1}]}`),
					Headers: jsonHeaders(ctJSON),
				})
				require.ErrorContains(t, err, "entry 1 is missing an embedding")
			})
		}
	})

	t.Run("wrong content type fails", func(t *testing.T) {
		t.Parallel()

		for _, tc := range singles {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				headers := http.Header{}
				headers.Set("Content-Type", "text/plain")

				_, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body:    []byte(`{"data":[{"embedding":[1]}]}`),
					Headers: headers,
				})
				require.ErrorContains(t, err, "expected Content-Type application/json")
			})
		}
	})

	t.Run("malformed body fails", func(t *testing.T) {
		t.Parallel()

		for _, tc := range batches {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				_, err := tc.codec.Decode(context.Background(), DecodeParams{
					Body:    []byte("{not json"),
					Headers: jsonHeaders(ctJSON),
				})
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
			})
		}
	})
}

// assertOpenAIEmbeddingsCodecEncode pins the encode behavior every provider
// codec family speaking the OpenAI embeddings wire format shares: the JSON
// body carries input(s) and the provider-side model identifier, provider
// parameters merge at the top level and cannot override the model, typed
// hf-inference feature-extraction parameters are not forwarded, a supplied
// mapping wins without a resolver call, nil provider parameters are skipped,
// and serialization/configuration failures surface as SDK errors.
// model is the request model the codec resolves through its resolver;
// wireModel is the provider-side model identifier expected in the JSON body
// (it differs from model whenever the provider remaps Hub model IDs).
// resolverCodec builds a codec from a mixin wired to the given resolver and
// backs the mapping/resolver interaction cases. Add a case per new provider
// adopting the format.
func assertOpenAIEmbeddingsCodecEncode(
	t *testing.T,
	codec FeatureExtractionPooledCodec,
	model string,
	wireModel string,
	resolverCodec func(ModelResolver) FeatureExtractionPooledCodec,
) {
	t.Helper()

	t.Run("openai-style payload", func(t *testing.T) {
		t.Parallel()

		body, headers, err := codec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{
					Input: "What is the capital of France?",
				},
				Model: model,
			},
		)
		require.NoError(t, err)
		require.Equal(t, "application/json", headers.Get("Content-Type"))
		require.Equal(t, "application/json", headers.Get("Accept"))
		require.JSONEq(
			t,
			`{"input":"What is the capital of France?","model":"`+wireModel+`"}`,
			string(body),
		)
	})

	t.Run("provider parameters merge and cannot override the model", func(t *testing.T) {
		t.Parallel()

		body, _, err := codec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{
					Input: "hello",
					ProviderParams: map[string]any{
						"dimensions":      256,
						"encoding_format": "float",
						"model":           "attacker-model",
					},
				},
				Model: model,
			},
		)
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"input":"hello","dimensions":256,"encoding_format":"float","model":"`+wireModel+`"}`,
			string(body),
		)
	})

	t.Run("typed hf parameters are not forwarded", func(t *testing.T) {
		t.Parallel()

		normalize := true
		body, _, err := codec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{
					Input:      "hello",
					Parameters: &hftypes.FeatureExtractionParameters{Normalize: &normalize},
				},
				Model: model,
			},
		)
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"input":"hello","model":"`+wireModel+`"}`,
			string(body),
		)
	})

	t.Run("supplied mapping wins without a resolver call", func(t *testing.T) {
		t.Parallel()

		resolver := &recordingResolver{}
		mixinCodec := resolverCodec(resolver)

		body, _, err := mixinCodec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{Input: "hello"},
				Model:   model,
				Mapping: &ModelMapping{ProviderModelID: "mapped-at-endpoint-construction"},
			},
		)
		require.NoError(t, err)
		require.Zero(t, resolver.calls.Load())
		require.JSONEq(
			t,
			`{"input":"hello","model":"mapped-at-endpoint-construction"}`,
			string(body),
		)
	})

	t.Run("unmarshalable provider parameter fails", func(t *testing.T) {
		t.Parallel()

		_, _, err := codec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{
					Input:          "hello",
					ProviderParams: map[string]any{"bad": make(chan int)},
				},
				Model: model,
			},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})

	t.Run("nil provider parameters are skipped", func(t *testing.T) {
		t.Parallel()

		body, _, err := codec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{
					Input:          "hello",
					ProviderParams: map[string]any{"ignored": nil},
				},
				Model: model,
			},
		)
		require.NoError(t, err)
		require.JSONEq(
			t,
			`{"input":"hello","model":"`+wireModel+`"}`,
			string(body),
		)
	})

	t.Run("missing mapping and resolver fails", func(t *testing.T) {
		t.Parallel()

		zeroCodec := resolverCodec(nil)

		_, _, err := zeroCodec.Encode(
			context.Background(),
			EncodeParams[hftypes.FeatureExtractionRequest]{
				Request: hftypes.FeatureExtractionRequest{Input: "hello"},
				Model:   model,
			},
		)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}
