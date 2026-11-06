//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// The SDK contract (see AGENTS.md, "Context Handling") is that a nil context
// falls back to context.Background(). These tests pin that contract end to
// end: calling through the public API with a nil ctx must neither panic nor
// fail, because the task layer normalizes the context at its boundary. They
// exercise the unary JSON, streaming, and media-response paths.

func TestNilContext_Chat(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
		hfopts.WithModel("test-model"),
	)

	//nolint:staticcheck // SA1012: nil ctx is deliberate — this test pins the documented nil-context fallback contract
	resp, err := client.Chat(nil, hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hi")}},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, mt.LastRequest, "request must be dispatched with the normalized context")
	require.NotNil(t, mt.LastRequest.Context(), "request must carry a context")
	require.NotEmpty(t, resp.Choices)
	require.NotEmpty(t, *resp.Choices[0].Message.Content)
}

func TestNilContext_ChatStream(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"id\",\"created\":1,\"model\":\"stream-model\",\"system_fingerprint\":\"sig\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
		hfopts.WithModel("test-model"),
	)

	//nolint:staticcheck // SA1012: nil ctx is deliberate — this test pins the documented nil-context fallback contract
	stream, err := client.ChatStream(nil, hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hi")}},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, stream)
	defer func() { _ = stream.Close() }()

	require.NotNil(t, mt.LastRequest, "request must be dispatched with the normalized context")
	require.NotNil(t, mt.LastRequest.Context(), "request must carry a context")
}

func TestNilContext_GenerateImage(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "\x89PNG\r\n\x1a\n", nil)
	mt.Response.Header.Set("Content-Type", "image/png")
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
		hfopts.WithModel("test-model"),
	)

	//nolint:staticcheck // SA1012: nil ctx is deliberate — this test pins the documented nil-context fallback contract
	resp, err := client.GenerateImage(nil, hftypes.TextToImageRequest{Input: "a cat"})
	require.NoError(t, err)
	require.NotNil(t, mt.LastRequest, "request must be dispatched with the normalized context")
	require.NotNil(t, mt.LastRequest.Context(), "request must carry a context")
	require.Equal(t, "image/png", resp.MediaType)
}
