//go:build !integration

package hfgo

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/providers"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/sdkversion"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

const chatServiceResponseBody = `{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`

type mockProvider struct {
	name string
}

func (p mockProvider) Endpoint(_ Task, _ string) (string, error) {
	return "", nil
}

func (p mockProvider) ProviderSuffix() string {
	return p.name
}

func TestNewClient_Defaults(t *testing.T) {
	t.Parallel()

	client := NewClient()

	require.Equal(t, request.DefaultBaseURL, client.opts.BaseURL)
	require.Equal(t, request.DefaultToken, client.opts.Token)
	require.Equal(t, request.DefaultModel, client.opts.Model)
	require.Equal(t, providers.HuggingFaceProvider{}, client.opts.Provider)
	require.Equal(t, request.DefaultMaxResponseBodyBytes, client.opts.MaxResponseBodyBytes)
	require.Equal(t, sdkversion.UserAgent(), client.opts.UserAgent)
	require.Nil(t, client.opts.Headers)
	require.NotNil(t, client.opts.HTTPClient)
}

func TestChatService_Complete_ModelSelection(t *testing.T) {
	t.Parallel()

	text := "hi"

	cases := []struct {
		name        string
		clientModel string
		optsModel   string
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
			optsModel:   "explicit-model",
			wantModel:   "explicit-model",
		},
		{
			name:        "respects request model",
			clientModel: "default-model",
			optsModel:   "opts-model",
			reqModel:    testutils.Ptr("explicit-model"),
			wantModel:   "explicit-model",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
			client := NewClient(
				WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				WithModel(tc.clientModel),
			)
			req := ChatRequest{
				Model: tc.reqModel,
				Messages: []ChatMessage{
					{Role: "user", Content: ChatMessageContent{Text: &text}},
				},
			}

			var err error
			if tc.optsModel != "" {
				_, err = client.Chat(req, WithModel(tc.optsModel))
			} else {
				_, err = client.Chat(req)
			}

			require.NoError(t, err)

			require.NotNil(t, mt.LastRequest)
			require.Equal(t, "/v1/chat/completions", mt.LastRequest.URL.Path)

			got := testutils.ReadRequestBody(t, mt)
			require.Equal(t, tc.wantModel, got["model"])

			if tc.reqModel == nil {
				require.Nil(t, req.Model)
			} else {
				require.NotNil(t, req.Model)
				require.Equal(t, *tc.reqModel, *req.Model)
			}
		})
	}
}

func TestChatService_Complete_ModelValidation(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	text := "hi"
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: ChatMessageContent{Text: &text}},
		},
	}

	_, err := client.Chat(req)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestChatService_Complete_ZeroValueRequest(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	_, err := client.Chat(ChatRequest{})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestChatService_Complete_StreamNotAllowed(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(http.StatusOK, chatServiceResponseBody, nil)
	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	text := "hi"
	stream := true
	req := ChatRequest{
		Stream: &stream,
		Messages: []ChatMessage{
			{Role: "user", Content: ChatMessageContent{Text: &text}},
		},
	}

	_, err := client.Chat(req)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestChatService_CompleteStream_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"id\",\"created\":1,\"model\":\"stream-model\",\"system_fingerprint\":\"sig\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		WithModel("default-model"),
	)

	text := "hi"
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: ChatMessageContent{Text: &text}},
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
	payload := testutils.ReadRequestBody(t, mt)
	require.Equal(t, true, payload["stream"])
	require.Equal(t, "default-model", payload["model"])
}

func TestChatStream_Recv_MergesToolCallMetadata(t *testing.T) {
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
	assertToolCallStream(t, body, func(chunks []ChatStreamResponse) {
		require.Len(t, chunks, 2)
		first, second := chunks[0], chunks[1]
		require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)
		require.Equal(t, "call_0", second.Choices[0].Delta.ToolCalls[0].ID)
		require.Equal(t, `{"foo":1}`, second.Choices[0].Delta.ToolCalls[0].Function.Arguments)
	})
}

func TestChatStream_Recv_MergesAcrossChoices(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"id":"id","created":1,"model":"stream-model","system_fingerprint":"sig","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_0","type":"function","index":0,"function":{"name":"fn","arguments":""}}]}},{"index":1,"delta":{"tool_calls":[{"id":"call_1","type":"function","index":0,"function":{"name":"fn2","arguments":""}}]}}]}`,
		``,
		`data: {"id":"id","created":1,"model":"stream-model","system_fingerprint":"sig","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"foo\":1}"}}]}},{"index":1,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"bar\":2}"}}]}}]}`,
		``,
		"data: [DONE]",
		``,
		``,
	}, "\n")

	assertToolCallStream(t, body, func(chunks []ChatStreamResponse) {
		require.Len(t, chunks, 2)
		first, second := chunks[0], chunks[1]
		require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)
		require.Equal(t, "call_1", first.Choices[1].Delta.ToolCalls[0].ID)
		require.Equal(t, "call_0", second.Choices[0].Delta.ToolCalls[0].ID)
		require.Equal(t, "call_1", second.Choices[1].Delta.ToolCalls[0].ID)
		require.Equal(t, `{"foo":1}`, second.Choices[0].Delta.ToolCalls[0].Function.Arguments)
		require.Equal(t, `{"bar":2}`, second.Choices[1].Delta.ToolCalls[0].Function.Arguments)
	})
}

func TestChatStream_Recv_InvalidJSONError(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"id":"id","created":1,"model":"stream-model","system_fingerprint":"sig","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_0","type":"function","index":0,"function":{"name":"fn","arguments":""}}]}}]}`,
		``,
		"data: {not json}",
		``,
	}, "\n")

	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		WithModel("default-model"),
	)

	text := "hi"
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: ChatMessageContent{Text: &text}},
		},
	}

	stream, err := client.ChatStream(req)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	first, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)

	_, err = stream.Recv(context.Background())
	require.Error(t, err)
}

// assertToolCallStream streams the provided SSE body through ChatStream and
// passes all decoded chunks to the supplied assertion callback.
func assertToolCallStream(
	t *testing.T,
	body string,
	assertions func(chunks []ChatStreamResponse),
) {
	t.Helper()

	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		WithModel("default-model"),
	)

	text := "hi"
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: ChatMessageContent{Text: &text}},
		},
	}

	stream, err := client.ChatStream(req)
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	var chunks []ChatStreamResponse
	for {
		chunk, err := stream.Recv(context.Background())
		if err != nil {
			require.ErrorIs(t, err, io.EOF)

			break
		}
		chunks = append(chunks, chunk)
	}
	assertions(chunks)
}

func TestChatService_CompleteStream_ZeroValueRequest(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "", nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")
	client := NewClient(
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	_, err := client.ChatStream(ChatRequest{})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	require.Nil(t, mt.LastRequest)
}

func TestApplyProvider(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		model       *string
		provider    Provider
		wantModel   *string
		description string
	}{
		{
			name:        "applies provider to model without provider",
			model:       testutils.Ptr("mistral-7b"),
			provider:    providers.HuggingFaceProvider{},
			wantModel:   testutils.Ptr("mistral-7b"),
			description: "model + provider → model:provider",
		},
		{
			name:        "ignores provider when model already has provider",
			model:       testutils.Ptr("mistral-7b:mistral"),
			provider:    providers.HuggingFaceProvider{},
			wantModel:   testutils.Ptr("mistral-7b:mistral"),
			description: "model:provider + different provider → unchanged",
		},
		{
			name:        "returns nil model when model is nil",
			model:       nil,
			provider:    providers.HuggingFaceProvider{},
			wantModel:   nil,
			description: "nil model → nil",
		},
		{
			name:        "returns nil model when model is empty string",
			model:       testutils.Ptr(""),
			provider:    providers.HuggingFaceProvider{},
			wantModel:   testutils.Ptr(""),
			description: "empty model → empty",
		},
		{
			name:        "returns model unchanged when provider is empty",
			model:       testutils.Ptr("mistral-7b"),
			provider:    nil,
			wantModel:   testutils.Ptr("mistral-7b"),
			description: "model + empty provider → model unchanged",
		},
		{
			name:        "handles provider with special characters",
			model:       testutils.Ptr("mistral-7b"),
			provider:    mockProvider{name: "provider-name"},
			wantModel:   testutils.Ptr("mistral-7b:provider-name"),
			description: "provider with hyphens",
		},
		{
			name:        "handles provider with underscores",
			model:       testutils.Ptr("mistral-7b"),
			provider:    mockProvider{name: "provider_name"},
			wantModel:   testutils.Ptr("mistral-7b:provider_name"),
			description: "provider with underscores",
		},
		{
			name:        "handles provider with dots",
			model:       testutils.Ptr("mistral-7b"),
			provider:    mockProvider{name: "provider.com"},
			wantModel:   testutils.Ptr("mistral-7b:provider.com"),
			description: "provider with dots",
		},
		{
			name:        "ignores provider when model has multiple colons",
			model:       testutils.Ptr("org:model:variant"),
			provider:    providers.HuggingFaceProvider{},
			wantModel:   testutils.Ptr("org:model:variant"),
			description: "model with multiple colons",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := applyProvider(tc.model, tc.provider)
			// Verify result matches expectation
			if tc.wantModel == nil {
				require.Nil(t, got, tc.description)

				return
			}
			require.NotNil(t, got, tc.description)
			require.Equal(t, *tc.wantModel, *got, tc.description)
		})
	}
}

func TestResolveModel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		reqModel       *string
		clientModel    string
		clientProvider Provider
		optsModel      string
		optsProvider   Provider
		wantModel      *string
		description    string
	}{
		{
			name:        "uses request model when provided",
			reqModel:    testutils.Ptr("request-model"),
			clientModel: "client-model",
			wantModel:   testutils.Ptr("request-model"),
			description: "request model takes precedence",
		},
		{
			name:        "uses options model when request model is nil",
			clientModel: "client-model",
			optsModel:   "opts-model",
			wantModel:   testutils.Ptr("opts-model"),
			description: "options model used when request is nil",
		},
		{
			name:        "uses client model when request and options are nil",
			clientModel: "client-model",
			wantModel:   testutils.Ptr("client-model"),
			description: "client model used as fallback",
		},
		{
			name:           "applies provider to resolved model",
			clientModel:    "mistral-7b",
			clientProvider: providers.HuggingFaceProvider{},
			wantModel:      testutils.Ptr("mistral-7b"),
			description:    "provider applied to client model",
		},
		{
			name:           "applies options provider to request model",
			reqModel:       testutils.Ptr("mistral-7b"),
			clientModel:    "client-model",
			clientProvider: mockProvider{name: "client-provider"},
			optsProvider:   mockProvider{name: "opts-provider"},
			wantModel:      testutils.Ptr("mistral-7b:opts-provider"),
			description:    "options provider applied to request model",
		},
		{
			name:         "request model with provider ignores provider option",
			reqModel:     testutils.Ptr("mistral-7b:mistral"),
			clientModel:  "client-model",
			optsProvider: providers.HuggingFaceProvider{},
			wantModel:    testutils.Ptr("mistral-7b:mistral"),
			description:  "existing provider in model not overridden",
		},
		{
			name:           "empty request model falls back to options model",
			reqModel:       testutils.Ptr(""),
			optsModel:      "opts-model",
			clientProvider: mockProvider{name: "client-provider"},
			optsProvider:   mockProvider{name: "opts-provider"},
			wantModel:      testutils.Ptr("opts-model:opts-provider"),
			description:    "empty string treated as nil for fallback",
		},
		{
			name:        "no provider applied when provider is empty",
			clientModel: "mistral-7b",
			wantModel:   testutils.Ptr("mistral-7b"),
			description: "model unchanged when no provider",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payload := &ChatRequest{Model: tc.reqModel}
			// Manually set client-level options by creating a base options and applying overrides
			baseOpts := request.NewOptions().
				WithModel(tc.clientModel).
				WithProvider(tc.clientProvider)

			// For this test, we'll apply the options with the client defaults
			optsOverride := baseOpts
			if tc.optsModel != "" {
				optsOverride = optsOverride.With(WithModel(tc.optsModel))
			}
			if tc.optsProvider != nil {
				optsOverride = optsOverride.With(WithProvider(tc.optsProvider))
			}

			resolveModel(payload, optsOverride)

			if tc.wantModel == nil {
				if payload.Model != nil {
					t.Fatalf("expected nil, got %#v", payload.Model)
				}
			} else {
				if payload.Model == nil {
					t.Fatalf("expected %#v, got nil", tc.wantModel)
				}
				if *payload.Model != *tc.wantModel {
					t.Fatalf("expected %#v, got %#v", *tc.wantModel, *payload.Model)
				}
			}
		})
	}
}

func TestChatService_ProviderFallback(t *testing.T) {
	t.Parallel()

	type TestCase struct {
		name           string
		clientModel    string
		clientProvider Provider
		optsModel      *string
		optsProvider   Provider
		reqModel       *string
		wantModel      string
		wantErr        bool
		description    string
	}

	cases := []TestCase{
		{
			name:           "happy path with default HF provider",
			clientModel:    "mistral-7b",
			clientProvider: providers.HuggingFaceProvider{},
			wantModel:      "mistral-7b",
			description:    "end-to-end chat with HF provider (no suffix)",
		},
		{
			name:           "happy path with model that already has provider",
			clientModel:    "mistral-7b",
			clientProvider: providers.HuggingFaceProvider{},
			reqModel:       testutils.Ptr("mistral-7b:sambanova"),
			wantModel:      "mistral-7b:sambanova",
			description:    "model with existing provider is not modified",
		},
		{
			name:           "error on nil provider",
			clientModel:    "mistral-7b",
			clientProvider: nil,
			wantErr:        true,
			description:    "nil provider returns configuration error",
		},
	}

	// testImpl is a helper that tests model and provider resolution
	// for both Complete and CompleteStream methods. It handles all setup, method invocation,
	// and verification of the resolved model in the request.
	testImpl := func(
		t *testing.T,
		tc TestCase,
		mtFactory func() *testutils.MockTransport,
		methodCall func(*Client, ChatRequest, []Option) error,
	) {
		t.Helper()

		mt := mtFactory()
		client := NewClient(
			WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
			WithModel(tc.clientModel),
			WithProvider(tc.clientProvider),
		)

		text := "hi"
		req := ChatRequest{
			Model: tc.reqModel,
			Messages: []ChatMessage{
				{Role: "user", Content: ChatMessageContent{Text: &text}},
			},
		}

		optsToPass := []Option{}
		if tc.optsModel != nil {
			optsToPass = append(optsToPass, WithModel(*tc.optsModel))
		}
		if tc.optsProvider != nil {
			optsToPass = append(optsToPass, WithProvider(tc.optsProvider))
		}

		err := methodCall(&client, req, optsToPass)
		if tc.wantErr {
			require.Error(t, err, tc.description)

			return
		}
		require.NoError(t, err)

		require.NotNil(t, mt.LastRequest)
		require.Equal(t, "/v1/chat/completions", mt.LastRequest.URL.Path)

		got := testutils.ReadRequestBody(t, mt)
		require.Equal(t, tc.wantModel, got["model"], tc.description)
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Test Complete method
			testImpl(t, tc,
				func() *testutils.MockTransport {
					return testutils.NewJSONMockTransport(
						http.StatusOK,
						chatServiceResponseBody,
						nil,
					)
				},
				func(client *Client, req ChatRequest, opts []Option) error {
					_, err := client.Chat(req, opts...)

					return err
				},
			)

			// Test CompleteStream method (skip for error cases)
			if !tc.wantErr {
				sseBody := "data: {\"id\":\"id\",\"created\":1,\"model\":\"" + tc.wantModel + "\",\"system_fingerprint\":\"sig\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
					"data: [DONE]\n\n"
				testImpl(t, tc,
					func() *testutils.MockTransport {
						mt := testutils.NewMockTransport(http.StatusOK, sseBody, nil)
						mt.Response.Header.Set("Content-Type", "text/event-stream")

						return mt
					},
					func(client *Client, req ChatRequest, opts []Option) error {
						stream, err := client.ChatStream(req, opts...)
						if err != nil {
							return err
						}
						defer func() { _ = stream.Close() }()

						chunk, err := stream.Recv(context.Background())
						if err != nil {
							return err
						}
						require.Equal(t, tc.wantModel, chunk.Model)

						return nil
					},
				)
			}
		})
	}
}
