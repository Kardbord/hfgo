//go:build !integration

package hfgo

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfgoversion"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const chatServiceResponseBody = `{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`

func TestNewClient_Defaults(t *testing.T) {
	t.Parallel()

	client := NewClient()

	require.Equal(t, hfopts.DefaultBaseURL, client.opts.BaseURL)
	require.Equal(t, hfopts.DefaultToken, client.opts.Token)
	require.Equal(t, hfopts.DefaultModel, client.opts.Model)
	require.Equal(t, hfproviders.NewHuggingFaceProvider(), client.opts.Provider)
	require.Equal(t, hfopts.DefaultMaxResponseBodyBytes, client.opts.MaxResponseBodyBytes)
	require.Equal(t, hfgoversion.UserAgent(), client.opts.UserAgent)
	require.Nil(t, client.opts.Headers)
	require.NotNil(t, client.opts.HTTPClient)
}

func TestClientChat_ModelSelection(t *testing.T) {
	t.Parallel()

	text := "hi"

	cases := []struct {
		name        string
		clientModel string
		optModel    string
		reqModel    *string
		wantModel   string
	}{
		{
			name:        "uses client model when request and opt model missing",
			clientModel: "default-model",
			wantModel:   "default-model",
		},
		{
			name:        "uses opt model when request missing",
			clientModel: "default-model",
			optModel:    "explicit-model",
			wantModel:   "explicit-model",
		},
		{
			name:        "respects request model",
			clientModel: "default-model",
			optModel:    "opts-model",
			reqModel:    testutils.Ptr("explicit-model"),
			wantModel:   "explicit-model",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
			client := NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel(tc.clientModel),
			)

			req := hftypes.ChatRequest{
				Model: tc.reqModel,
				Messages: []hftypes.ChatMessage{
					{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
				},
			}

			var err error
			if tc.optModel != "" {
				_, err = client.Chat(req, hfopts.WithModel(tc.optModel))
			} else {
				_, err = client.Chat(req)
			}
			require.NoError(t, err)

			require.NotNil(t, mt.LastRequest)
			require.Equal(t, "/v1/chat/completions", mt.LastRequest.URL.Path)

			got := testutils.ReadRequestBody(t, mt)
			require.Equal(t, tc.wantModel, got["model"])
		})
	}
}

func TestClientChat_MissingModel(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	text := "hi"
	req := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
		},
	}

	_, err := client.Chat(req)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestClientChat_StreamNotAllowed(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("default-model"),
	)

	stream := true
	text := "hi"
	req := hftypes.ChatRequest{
		Stream: &stream,
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
		},
	}

	_, err := client.Chat(req)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestClientChatStream_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"id\",\"created\":1,\"model\":\"stream-model\",\"system_fingerprint\":\"sig\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("default-model"),
	)

	text := "hi"
	req := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
		},
	}

	stream, err := client.ChatStream(req)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "stream-model", chunk.Model)
	require.Len(t, chunk.Choices, 1)
	require.NotNil(t, chunk.Choices[0].Delta.Content)
	require.Equal(t, "hi", *chunk.Choices[0].Delta.Content)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)

	require.NotNil(t, mt.LastRequest)
	require.Equal(t, "/v1/chat/completions", mt.LastRequest.URL.Path)

	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, true, payload["stream"])
	require.Equal(t, "default-model", payload["model"])
}

func TestClientChatStream_MissingModel(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "", nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")
	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	_, err := client.ChatStream(hftypes.ChatRequest{})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestClientChatStream_ToolCallMerging(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"id":"id","created":1,"model":"stream-model","system_fingerprint":"sig","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_0","type":"function","index":0,"function":{"name":"fn","arguments":""}}]}}]}`,
		``,
		`data: {"id":"id","created":1,"model":"stream-model","system_fingerprint":"sig","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"foo\":1}"}}]}}]}`,
		``,
		"data: [DONE]",
		``,
		``,
	}, "\n")

	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("default-model"),
	)

	text := "hi"
	req := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
		},
	}

	stream, err := client.ChatStream(req)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	first, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)

	second, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", second.Choices[0].Delta.ToolCalls[0].ID)
	require.Equal(t, `{"foo":1}`, second.Choices[0].Delta.ToolCalls[0].Function.Arguments)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
}

func TestClientChat_ProviderSuffix(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("mistral-7b"),
		hfopts.WithProvider(testutils.NewMockProvider("sambanova", "sambanova")),
	)

	text := "hi"
	req := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
		},
	}

	_, err := client.Chat(req)
	require.NoError(t, err)

	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, "mistral-7b:sambanova", payload["model"])
}

func TestClientDetectObjects(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"label":"person","score":0.95,"box":{"xmin":1,"ymin":2,"xmax":3,"ymax":4}}]`,
		nil,
	)
	client := NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	const img = testutils.TinyPNGBase64

	objects, err := client.DetectObjects(hftypes.ObjectDetectionRequest{Input: img})
	require.NoError(t, err)
	require.Len(t, objects, 1)
	require.Equal(t, "person", objects[0].Label)
	require.InEpsilon(t, 0.95, objects[0].Score, 0.001)
	require.Equal(
		t,
		hftypes.ObjectDetectionBoundingBox{XMin: 1, XMax: 3, YMin: 2, YMax: 4},
		objects[0].Box,
	)

	require.NotNil(t, mt.LastRequest)
	require.Equal(t, "/hf-inference/models/test-model", mt.LastRequest.URL.Path)

	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, img, payload["inputs"])
}
