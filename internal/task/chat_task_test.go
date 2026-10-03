//go:build !integration

package task

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

type mockProvider struct {
	hfproviders.DefaultCodec

	name string
}

func (p mockProvider) Endpoint(_ hfproviders.Task, _ string) (string, error) {
	return "", nil
}

func (p mockProvider) ProviderSuffix() string {
	return p.name
}

func TestApplyProvider(t *testing.T) {
	t.Parallel()

	t.Run("applies provider suffix", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		provider := mockProvider{name: "sambanova"}
		want := "mistral-7b:sambanova"
		got := applyProvider(model, provider)
		require.NotNil(t, got)
		require.Equal(t, want, *got)
	})

	t.Run("ignores provider with empty suffix", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		got := applyProvider(model, hfproviders.NewHuggingFaceProvider())
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("returns nil when model is nil", func(t *testing.T) {
		got := applyProvider(nil, mockProvider{name: "sambanova"})
		require.Nil(t, got)
	})

	t.Run("returns model unchanged when model is empty", func(t *testing.T) {
		model := testutils.Ptr("")
		got := applyProvider(model, mockProvider{name: "sambanova"})
		require.NotNil(t, got)
		require.Empty(t, *got)
	})

	t.Run("returns model unchanged when provider is nil", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		got := applyProvider(model, nil)
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("ignores provider when model already has suffix", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b:mistral")
		got := applyProvider(model, mockProvider{name: "sambanova"})
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b:mistral", *got)
	})

	t.Run("ignores provider when model has multiple colons", func(t *testing.T) {
		model := testutils.Ptr("org:model:variant")
		got := applyProvider(model, mockProvider{name: "sambanova"})
		require.NotNil(t, got)
		require.Equal(t, "org:model:variant", *got)
	})

	t.Run("handles provider with special characters", func(t *testing.T) {
		model := testutils.Ptr("mistral-7b")
		got := applyProvider(model, mockProvider{name: "provider-name_v1.0"})
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b:provider-name_v1.0", *got)
	})
}

func TestResolveModel(t *testing.T) {
	t.Parallel()

	t.Run("uses request model when provided", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: testutils.Ptr("request-model")}
		resolveModel(payload, hfopts.NewOptions().With(hfopts.WithModel("client-model")))
		require.NotNil(t, payload.Model)
		require.Equal(t, "request-model", *payload.Model)
	})

	t.Run("uses options model when request model is nil", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		resolveModel(payload, hfopts.NewOptions().With(hfopts.WithModel("opts-model")))
		require.NotNil(t, payload.Model)
		require.Equal(t, "opts-model", *payload.Model)
	})

	t.Run("uses options model when request model is empty", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: testutils.Ptr("")}
		resolveModel(payload, hfopts.NewOptions().With(hfopts.WithModel("opts-model")))
		require.NotNil(t, payload.Model)
		require.Equal(t, "opts-model", *payload.Model)
	})

	t.Run("applies provider suffix to resolved model", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("mistral-7b"),
			hfopts.WithProvider(mockProvider{name: "sambanova"}),
		))
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b:sambanova", *payload.Model)
	})

	t.Run("request model with suffix ignores provider", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: testutils.Ptr("mistral-7b:mistral")}
		resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("client-model"),
			hfopts.WithProvider(mockProvider{name: "sambanova"}),
		))
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b:mistral", *payload.Model)
	})

	t.Run("HF provider does not append suffix", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("mistral-7b"),
			hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()),
		))
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b", *payload.Model)
	})
}

func TestResolveChatOptions(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is missing", func(t *testing.T) {
		opts := hfopts.NewOptions().With(hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()))
		_, err := resolveChatOptions(opts, &hftypes.ChatRequest{})
		require.Error(t, err)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		opts := hfopts.NewOptions().With(hfopts.WithProvider(nil))
		_, err := resolveChatOptions(opts, &hftypes.ChatRequest{Model: testutils.Ptr("model")})
		require.Error(t, err)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("sets resolved model on options", func(t *testing.T) {
		opts := hfopts.NewOptions().With(
			hfopts.WithModel("client-model"),
			hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()),
		)
		updated, err := resolveChatOptions(opts, &hftypes.ChatRequest{
			Model: testutils.Ptr("request-model"),
		})
		require.NoError(t, err)
		require.Equal(t, "request-model", updated.Model)
	})
}

const chatServiceResponseBody = `{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`

func TestChat_RejectStream(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("default-model"),
		hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()),
	)

	stream := true
	_, err := Chat(opts, hftypes.ChatRequest{
		Model: testutils.Ptr("request-model"),
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: testutils.Ptr("hi")}},
		},
		Stream: &stream,
	})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestStreamChat_SetsStreamTrue(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"id\",\"created\":1,\"model\":\"stream-model\",\"system_fingerprint\":\"sig\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("default-model"),
		hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()),
	)

	stream, err := StreamChat(opts, hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: testutils.Ptr("hi")}},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, stream)
	require.NoError(t, stream.Close())

	require.NotNil(t, mt.LastRequest)
	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, true, payload["stream"])
	require.Equal(t, "default-model", payload["model"])
}
