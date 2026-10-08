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
			reqModel:    new("explicit-model"),
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

func TestClient_EndpointDelegates(t *testing.T) {
	t.Parallel()

	question := hftypes.QuestionAnsweringInput{
		Question: "What is the capital of France?",
		Context:  "France is a country in Europe. Its capital is Paris.",
	}
	table := hftypes.TableQuestionAnsweringInput{
		Question: "How old is Bob?",
		Table: map[string][]string{
			"Name": {"Alice", "Bob", "Carol"},
			"Age":  {"25", "30", "35"},
		},
	}

	cases := []struct {
		name     string
		response string
		call     func(client Client) error
	}{
		{
			name:     "ClassifyText",
			response: `[[{"label":"positive","score":0.95}]]`,
			call: func(client Client) error {
				_, err := client.ClassifyText(hftypes.TextClassificationRequest{Input: "test text"})

				return err
			},
		},
		{
			name:     "ClassifyTextBatch",
			response: `[[{"label":"positive","score":0.95}]]`,
			call: func(client Client) error {
				_, err := client.ClassifyTextBatch(
					hftypes.TextClassificationBatchRequest{Inputs: []string{"test text"}},
				)

				return err
			},
		},
		{
			name:     "ClassifyTokens",
			response: `[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
			call: func(client Client) error {
				_, err := client.ClassifyTokens(
					hftypes.TokenClassificationRequest{Input: "My name is Sarah."},
				)

				return err
			},
		},
		{
			name:     "ClassifyTokensBatch",
			response: `[[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]]`,
			call: func(client Client) error {
				_, err := client.ClassifyTokensBatch(
					hftypes.TokenClassificationBatchRequest{Inputs: []string{"My name is Sarah."}},
				)

				return err
			},
		},
		{
			name:     "AnswerQuestion",
			response: `{"answer":"Paris","score":0.95,"start":48,"end":53}`,
			call: func(client Client) error {
				_, err := client.AnswerQuestion(hftypes.QuestionAnsweringRequest{Input: question})

				return err
			},
		},
		{
			name:     "ZeroShotClassifyText",
			response: `[{"label":"positive","score":0.95}]`,
			call: func(client Client) error {
				_, err := client.ZeroShotClassifyText(hftypes.ZeroShotTextClassificationRequest{
					Input: "test text",
					Parameters: &hftypes.ZeroShotTextClassificationParameters{
						CandidateLabels: []string{"positive", "negative"},
					},
				})

				return err
			},
		},
		{
			name:     "ZeroShotClassifyTextBatch",
			response: `[{"Sequence":"test text","Labels":["positive","negative"],"Scores":[0.95,0.05]}]`,
			call: func(client Client) error {
				_, err := client.ZeroShotClassifyTextBatch(
					hftypes.ZeroShotTextClassificationBatchRequest{
						Inputs: []string{"test text"},
						Parameters: &hftypes.ZeroShotTextClassificationParameters{
							CandidateLabels: []string{"positive", "negative"},
						},
					},
				)

				return err
			},
		},
		{
			name:     "FillMask",
			response: `[{"sequence":"The capital of France is Paris.","score":0.95,"token":1,"token_str":"Paris"}]`,
			call: func(client Client) error {
				_, err := client.FillMask(
					hftypes.FillMaskRequest{Input: "The capital of France is [MASK]."},
				)

				return err
			},
		},
		{
			name:     "FillMaskBatch",
			response: `[[{"sequence":"I walk my dog everyday.","score":0.95,"token":1,"token_str":"walk"}]]`,
			call: func(client Client) error {
				_, err := client.FillMaskBatch(
					hftypes.FillMaskBatchRequest{Inputs: []string{"I [MASK] my dog everyday."}},
				)

				return err
			},
		},
		{
			name:     "Summarize",
			response: `[{"summary_text":"A concise summary."}]`,
			call: func(client Client) error {
				_, err := client.Summarize(hftypes.SummarizationRequest{Input: "Some long text."})

				return err
			},
		},
		{
			name:     "SummarizeBatch",
			response: `[{"summary_text":"Summary one."}]`,
			call: func(client Client) error {
				_, err := client.SummarizeBatch(
					hftypes.SummarizationBatchRequest{Inputs: []string{"Long text one."}},
				)

				return err
			},
		},
		{
			name:     "AnswerTableQuestion",
			response: `{"answer":"30","cells":["30"],"coordinates":[[1,1]]}`,
			call: func(client Client) error {
				_, err := client.AnswerTableQuestion(
					hftypes.TableQuestionAnsweringRequest{Input: table},
				)

				return err
			},
		},
		{
			name:     "Translate",
			response: `[{"translation_text":"Bonjour le monde."}]`,
			call: func(client Client) error {
				_, err := client.Translate(hftypes.TranslationRequest{Input: "Hello world."})

				return err
			},
		},
		{
			name:     "TranslateBatch",
			response: `[{"translation_text":"Bonjour le monde."}]`,
			call: func(client Client) error {
				_, err := client.TranslateBatch(
					hftypes.TranslationBatchRequest{Inputs: []string{"Hello world."}},
				)

				return err
			},
		},
		{
			name:     "FeatureExtract",
			response: `[0.1,0.2,0.3]`,
			call: func(client Client) error {
				_, err := client.FeatureExtract(
					hftypes.FeatureExtractionRequest{Input: "hello"},
				)

				return err
			},
		},
		{
			name:     "FeatureExtractBatch",
			response: `[[0.1,0.2,0.3]]`,
			call: func(client Client) error {
				_, err := client.FeatureExtractBatch(
					hftypes.FeatureExtractionBatchRequest{Inputs: []string{"hello"}},
				)

				return err
			},
		},
		{
			name:     "DetectObjects",
			response: `[{"label":"person","score":0.95,"box":{"xmin":1,"ymin":2,"xmax":3,"ymax":4}}]`,
			call: func(client Client) error {
				_, err := client.DetectObjects(
					hftypes.ObjectDetectionRequest{Input: testutils.TinyPNGBase64},
				)

				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.response, nil)
			client := NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			require.NoError(t, tc.call(client))
			require.NotNil(t, mt.LastRequest)
			require.Contains(t, mt.LastRequest.URL.Path, "test-model")
		})
	}
}
