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

func TestApplyProvider(t *testing.T) {
	t.Parallel()

	t.Run("applies provider suffix", func(t *testing.T) {
		model := new("mistral-7b")
		provider := testutils.NewMockProvider("sambanova", "sambanova")
		want := "mistral-7b:sambanova"
		got := applyProvider(model, provider)
		require.NotNil(t, got)
		require.Equal(t, want, *got)
	})

	t.Run("ignores provider with empty suffix", func(t *testing.T) {
		model := new("mistral-7b")
		got := applyProvider(model, hfproviders.NewHuggingFaceProvider())
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("returns nil when model is nil", func(t *testing.T) {
		got := applyProvider(nil, testutils.NewMockProvider("sambanova", "sambanova"))
		require.Nil(t, got)
	})

	t.Run("returns model unchanged when model is empty", func(t *testing.T) {
		model := new("")
		got := applyProvider(model, testutils.NewMockProvider("sambanova", "sambanova"))
		require.NotNil(t, got)
		require.Empty(t, *got)
	})

	t.Run("returns model unchanged when provider is nil", func(t *testing.T) {
		model := new("mistral-7b")
		got := applyProvider(model, nil)
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("returns model unchanged when provider is typed nil", func(t *testing.T) {
		model := new("mistral-7b")
		var provider *hfproviders.HuggingFaceProvider
		got := applyProvider(model, provider)
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b", *got)
	})

	t.Run("ignores provider when model already has suffix", func(t *testing.T) {
		model := new("mistral-7b:mistral")
		got := applyProvider(model, testutils.NewMockProvider("sambanova", "sambanova"))
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b:mistral", *got)
	})

	t.Run("ignores provider when model has multiple colons", func(t *testing.T) {
		model := new("org:model:variant")
		got := applyProvider(model, testutils.NewMockProvider("sambanova", "sambanova"))
		require.NotNil(t, got)
		require.Equal(t, "org:model:variant", *got)
	})

	t.Run("handles provider with special characters", func(t *testing.T) {
		model := new("mistral-7b")
		got := applyProvider(
			model,
			testutils.NewMockProvider("provider-name_v1.0", "provider-name_v1.0"),
		)
		require.NotNil(t, got)
		require.Equal(t, "mistral-7b:provider-name_v1.0", *got)
	})
}

func TestResolveModel(t *testing.T) {
	t.Parallel()

	t.Run("uses request model when provided", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: new("request-model")}
		model, err := resolveModel(
			payload,
			hfopts.NewOptions().With(hfopts.WithModel("client-model")),
		)
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "request-model", *payload.Model)
		require.Equal(t, "request-model", model)
	})

	t.Run("uses options model when request model is nil", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		model, err := resolveModel(
			payload,
			hfopts.NewOptions().With(hfopts.WithModel("opts-model")),
		)
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "opts-model", *payload.Model)
		require.Equal(t, "opts-model", model)
	})

	t.Run("uses options model when request model is empty", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: new("")}
		model, err := resolveModel(
			payload,
			hfopts.NewOptions().With(hfopts.WithModel("opts-model")),
		)
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "opts-model", *payload.Model)
		require.Equal(t, "opts-model", model)
	})

	t.Run("applies provider suffix to resolved model", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		model, err := resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("mistral-7b"),
			hfopts.WithProvider(testutils.NewMockProvider("sambanova", "sambanova")),
		))
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b:sambanova", *payload.Model)
		require.Equal(t, "mistral-7b:sambanova", model)
	})

	t.Run("request model with suffix ignores provider", func(t *testing.T) {
		payload := &hftypes.ChatRequest{Model: new("mistral-7b:mistral")}
		model, err := resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("client-model"),
			hfopts.WithProvider(testutils.NewMockProvider("sambanova", "sambanova")),
		))
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b:mistral", *payload.Model)
		require.Equal(t, "mistral-7b:mistral", model)
	})

	t.Run("HF provider does not append suffix", func(t *testing.T) {
		payload := &hftypes.ChatRequest{}
		model, err := resolveModel(payload, hfopts.NewOptions().With(
			hfopts.WithModel("mistral-7b"),
			hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()),
		))
		require.NoError(t, err)
		require.NotNil(t, payload.Model)
		require.Equal(t, "mistral-7b", *payload.Model)
		require.Equal(t, "mistral-7b", model)
	})
}

func TestChat_Validation(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is missing", func(t *testing.T) {
		opts := hfopts.NewOptions().With(hfopts.WithProvider(hfproviders.NewHuggingFaceProvider()))
		_, err := Chat(opts, hftypes.ChatRequest{})
		require.Error(t, err)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		opts := hfopts.NewOptions().With(hfopts.WithProvider(nil))
		_, err := Chat(opts, hftypes.ChatRequest{Model: new("model")})
		require.Error(t, err)
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
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
		Model: new("request-model"),
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hi")}},
		},
		Stream: &stream,
	})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestChatStream_SetsStreamTrue(t *testing.T) {
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

	stream, err := ChatStream(opts, hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hi")}},
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
