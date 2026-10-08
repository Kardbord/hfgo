//go:build !integration

package hftypes_test

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestChatMessageContent_Marshal(t *testing.T) {
	t.Parallel()

	text := "hello"
	imgURL := "https://example.com/image.png"

	cases := []struct {
		name        string
		value       hftypes.ChatMessageContent
		wantJSON    string
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name:     "text",
			value:    hftypes.ChatMessageContent{Text: &text},
			wantJSON: `"hello"`,
		},
		{
			name:     "empty",
			value:    hftypes.ChatMessageContent{},
			wantJSON: `null`,
		},
		{
			name: "chunks",
			value: hftypes.ChatMessageContent{
				Chunks: []hftypes.ChatMessageChunk{
					{
						Type:     hftypes.MessageChunkTypeImageURL,
						ImageURL: &hftypes.ChatImageURL{URL: imgURL},
					},
				},
			},
			wantJSON: `[{"image_url":{"url":"https://example.com/image.png"},"type":"image_url"}]`,
		},
		{
			name: "both",
			value: hftypes.ChatMessageContent{
				Text: &text,
				Chunks: []hftypes.ChatMessageChunk{
					{Type: hftypes.MessageChunkTypeText, Text: &text},
				},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
			if tc.wantJSON != "" && string(data) != tc.wantJSON {
				t.Fatalf("unexpected json: %s", string(data))
			}
		})
	}
}

func TestChatMessageChunk_Validation(t *testing.T) {
	t.Parallel()

	text := "hello"

	cases := []struct {
		name        string
		value       hftypes.ChatMessageChunk
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name:  "text",
			value: hftypes.ChatMessageChunk{Type: hftypes.MessageChunkTypeText, Text: &text},
		},
		{
			name: "image_url",
			value: hftypes.ChatMessageChunk{
				Type:     hftypes.MessageChunkTypeImageURL,
				ImageURL: &hftypes.ChatImageURL{URL: "x"},
			},
		},
		{
			name:        "missing text",
			value:       hftypes.ChatMessageChunk{Type: hftypes.MessageChunkTypeText},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "missing image",
			value:       hftypes.ChatMessageChunk{Type: hftypes.MessageChunkTypeImageURL},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "missing type",
			value:       hftypes.ChatMessageChunk{},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "invalid type",
			value:       hftypes.ChatMessageChunk{Type: hftypes.MessageChunkType("other")},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "text with image_url",
			value: hftypes.ChatMessageChunk{
				Type:     hftypes.MessageChunkTypeText,
				Text:     &text,
				ImageURL: &hftypes.ChatImageURL{URL: "x"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "image_url with text",
			value: hftypes.ChatMessageChunk{
				Type:     hftypes.MessageChunkTypeImageURL,
				Text:     &text,
				ImageURL: &hftypes.ChatImageURL{URL: "x"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatImageURL_Validation(t *testing.T) {
	t.Parallel()

	_, err := json.Marshal(hftypes.ChatImageURL{})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestChatMessage_Validation(t *testing.T) {
	t.Parallel()

	text := "hi"

	cases := []struct {
		name        string
		value       hftypes.ChatMessage
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name: "content",
			value: hftypes.ChatMessage{
				Role:    "user",
				Content: hftypes.ChatMessageContent{Text: &text},
			},
		},
		{
			name: "tool_calls",
			value: hftypes.ChatMessage{Role: "assistant", ToolCalls: []hftypes.ChatToolCall{{
				ID:       "id",
				Type:     "function",
				Function: hftypes.ChatFunctionCall{Name: "do", Arguments: "{}"},
			}}},
		},
		{
			name: "both",
			value: hftypes.ChatMessage{
				Role:    "assistant",
				Content: hftypes.ChatMessageContent{Text: &text},
				ToolCalls: []hftypes.ChatToolCall{
					{
						ID:       "id",
						Type:     "function",
						Function: hftypes.ChatFunctionCall{Name: "do", Arguments: "{}"},
					},
				},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "neither",
			value:       hftypes.ChatMessage{Role: "assistant"},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "missing role",
			value:       hftypes.ChatMessage{Content: hftypes.ChatMessageContent{Text: &text}},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatRequestClone_Deep_Scalars(t *testing.T) {
	t.Parallel()

	req := &hftypes.ChatRequest{
		Model:            new("m"),
		FrequencyPenalty: new(0.5),
		LogProbs:         new(true),
		MaxTokens:        new(100),
		PresencePenalty:  new(-0.2),
		Seed:             new(int64(42)),
		Stop:             []string{"x"},
		Stream:           new(false),
		Temperature:      new(1.0),
		ToolPrompt:       new("prompt"),
		TopLogProbs:      new(1),
		TopP:             new(0.9),
	}

	cloned := req.Clone()

	*cloned.Model = "m2"
	*cloned.FrequencyPenalty = 0.9
	*cloned.LogProbs = false
	*cloned.MaxTokens = 200
	*cloned.PresencePenalty = 0.5
	*cloned.Seed = 7
	cloned.Stop[0] = "y"
	*cloned.Stream = true
	*cloned.Temperature = 2.0
	*cloned.ToolPrompt = "changed"
	*cloned.TopLogProbs = 5
	*cloned.TopP = 0.1

	require.Equal(t, "m", *req.Model)
	require.InEpsilon(t, 0.5, *req.FrequencyPenalty, 0.001)
	require.True(t, *req.LogProbs)
	require.Equal(t, 100, *req.MaxTokens)
	require.InEpsilon(t, -0.2, *req.PresencePenalty, 0.001)
	require.Equal(t, int64(42), *req.Seed)
	require.Equal(t, []string{"x"}, req.Stop)
	require.False(t, *req.Stream)
	require.InEpsilon(t, 1.0, *req.Temperature, 0.001)
	require.Equal(t, "prompt", *req.ToolPrompt)
	require.Equal(t, 1, *req.TopLogProbs)
	require.InEpsilon(t, 0.9, *req.TopP, 0.001)
}

func TestChatRequestClone_Deep_Nested(t *testing.T) {
	t.Parallel()

	req := &hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{Role: "assistant", ToolCalls: []hftypes.ChatToolCall{
				{
					ID:   "id1",
					Type: "function",
					Function: hftypes.ChatFunctionCall{
						Name:        "fn",
						Arguments:   "{}",
						Description: new("desc"),
					},
				},
			}},
			{Role: "user", Content: hftypes.ChatMessageContent{Chunks: []hftypes.ChatMessageChunk{
				{
					Type:     hftypes.MessageChunkTypeImageURL,
					ImageURL: &hftypes.ChatImageURL{URL: "https://example.com/img.png"},
				},
				{Type: hftypes.MessageChunkTypeText, Text: new("hi")},
			}}},
		},
		ResponseFormat: &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{
				Name:        "n",
				Description: new("d"),
				Strict:      new(true),
			},
		},
		StreamOptions: &hftypes.ChatStreamOptions{IncludeUsage: new(true)},
		ToolChoice:    &hftypes.ChatToolChoice{Function: &hftypes.ChatFunctionName{Name: "fn"}},
		Tools: []hftypes.ChatTool{
			{
				Type: "function",
				Function: hftypes.ChatFunctionDefinition{
					Name:        "f",
					Description: new("desc"),
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		},
	}

	cloned := req.Clone()

	*cloned.Messages[0].ToolCalls[0].Function.Description = "changed"
	cloned.Messages[1].Content.Chunks[0].ImageURL.URL = "https://example.com/other.png"
	*cloned.Messages[1].Content.Chunks[1].Text = "changed"
	*cloned.ResponseFormat.JSONSchema.Strict = false
	*cloned.StreamOptions.IncludeUsage = false
	cloned.ToolChoice.Function.Name = "other"
	cloned.Tools[0].Function.Parameters = json.RawMessage(`{"type":"object"}zzz`)

	require.Equal(t, "desc", *req.Messages[0].ToolCalls[0].Function.Description)
	require.Equal(t, "https://example.com/img.png", req.Messages[1].Content.Chunks[0].ImageURL.URL)
	require.Equal(t, "hi", *req.Messages[1].Content.Chunks[1].Text)
	require.True(t, *req.ResponseFormat.JSONSchema.Strict)
	require.True(t, *req.StreamOptions.IncludeUsage)
	require.Equal(t, "fn", req.ToolChoice.Function.Name)
	require.JSONEq(t, `{"type":"object"}`, string(req.Tools[0].Function.Parameters))
}

func TestChatRequestClone_Deep_JSONSchema(t *testing.T) {
	t.Parallel()

	schema := json.RawMessage(`{"type":"object"}`)
	req := &hftypes.ChatRequest{
		Model: new("m"),
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: new("hello")}},
		},
		Stop: []string{"x"},
		Tools: []hftypes.ChatTool{
			{
				Type: "function",
				Function: hftypes.ChatFunctionDefinition{
					Name:       "f",
					Parameters: schema,
				},
			},
		},
		ResponseFormat: &hftypes.ChatResponseFormat{
			Type:       hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "n", Schema: schema},
		},
		ToolChoice: &hftypes.ChatToolChoice{Mode: new(hftypes.ToolChoiceModeAuto)},
	}

	cloned := req.Clone()

	*cloned.Messages[0].Content.Text = "changed"
	cloned.Messages[0].Role = "assistant"
	cloned.Stop[0] = "y"
	cloned.Tools[0].Function.Parameters = json.RawMessage(`{"type":"object"}zzz`)
	cloned.ResponseFormat.JSONSchema.Schema = json.RawMessage(`{"type":"object"}xxx`)
	*cloned.ToolChoice.Mode = hftypes.ToolChoiceModeNone

	require.Equal(t, "hello", *req.Messages[0].Content.Text)
	require.Equal(t, "user", req.Messages[0].Role)
	require.Equal(t, []string{"x"}, req.Stop)
	require.JSONEq(t, `{"type":"object"}`, string(req.Tools[0].Function.Parameters))
	require.JSONEq(t, `{"type":"object"}`, string(req.ResponseFormat.JSONSchema.Schema))
	require.Equal(t, hftypes.ToolChoiceModeAuto, *req.ToolChoice.Mode)
}

func TestChatRequestClone_Nil(t *testing.T) {
	t.Parallel()

	var req *hftypes.ChatRequest
	require.Empty(t, req.Clone())

	var m *hftypes.ChatMessage
	require.Empty(t, m.Clone())

	var c *hftypes.ChatMessageContent
	require.Empty(t, c.Clone())

	var chunk *hftypes.ChatMessageChunk
	require.Empty(t, chunk.Clone())

	var u *hftypes.ChatImageURL
	require.Empty(t, u.Clone())

	var tc *hftypes.ChatToolCall
	require.Empty(t, tc.Clone())

	var fd *hftypes.ChatFunctionDefinition
	require.Empty(t, fd.Clone())

	var rf *hftypes.ChatResponseFormat
	require.Empty(t, rf.Clone())

	var cfg *hftypes.ChatJSONSchemaConfig
	require.Empty(t, cfg.Clone())

	var so *hftypes.ChatStreamOptions
	require.Empty(t, so.Clone())

	var tool *hftypes.ChatTool
	require.Empty(t, tool.Clone())

	var choice *hftypes.ChatToolChoice
	require.Empty(t, choice.Clone())

	var name *hftypes.ChatFunctionName
	require.Empty(t, name.Clone())
}

func TestChatRequest_MarshalSuccess(t *testing.T) {
	t.Parallel()

	text := "hi"
	imgURL := "https://example.com/image.png"
	model := "model"
	req := hftypes.ChatRequest{
		Model: &model,
		Messages: []hftypes.ChatMessage{
			{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Chunks: []hftypes.ChatMessageChunk{
						{
							Type:     hftypes.MessageChunkTypeImageURL,
							ImageURL: &hftypes.ChatImageURL{URL: imgURL},
						},
					},
				},
			},
		},
	}

	data, err := json.Marshal(req)
	require.NoError(t, err)

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unexpected json: %v", err)
	}
	if got["model"] != model {
		t.Fatalf("unexpected model: %#v", got["model"])
	}
	messages, ok := got["messages"].([]any)
	if !ok || len(messages) != 2 {
		t.Fatalf("unexpected messages: %#v", got["messages"])
	}
	first, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected first message: %#v", messages[0])
	}
	if first["role"] != "user" || first["content"] != "hi" {
		t.Fatalf("unexpected first message fields: %#v", first)
	}
	second, ok := messages[1].(map[string]any)
	if !ok {
		t.Fatalf("unexpected second message: %#v", messages[1])
	}
	if second["role"] != "user" {
		t.Fatalf("unexpected second message role: %#v", second["role"])
	}
	chunks, ok := second["content"].([]any)
	if !ok || len(chunks) != 1 {
		t.Fatalf("unexpected content chunks: %#v", second["content"])
	}
	chunk, ok := chunks[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected chunk: %#v", chunks[0])
	}
	if chunk["type"] != string(hftypes.MessageChunkTypeImageURL) {
		t.Fatalf("unexpected chunk type: %#v", chunk["type"])
	}
	imageURL, ok := chunk["image_url"].(map[string]any)
	if !ok || imageURL["url"] != imgURL {
		t.Fatalf("unexpected image_url: %#v", chunk["image_url"])
	}
}

func TestChatRequest_MarshalValidation(t *testing.T) {
	t.Parallel()

	text := "hi"
	model := "model"

	cases := []*struct {
		name        string
		value       hftypes.ChatRequest
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name: "missing model",
			value: hftypes.ChatRequest{
				Messages: []hftypes.ChatMessage{
					{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
				},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "empty model",
			value: hftypes.ChatRequest{
				Model: new(""),
				Messages: []hftypes.ChatMessage{
					{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
				},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "missing messages",
			value: hftypes.ChatRequest{
				Model: &model,
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "valid request",
			value: hftypes.ChatRequest{
				Model: &model,
				Messages: []hftypes.ChatMessage{
					{Role: "user", Content: hftypes.ChatMessageContent{Text: &text}},
				},
			},
			wantErr: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestChatToolChoice_Marshal(t *testing.T) {
	t.Parallel()

	mode := hftypes.ToolChoiceMode("provider-mode")

	cases := []struct {
		name        string
		value       hftypes.ChatToolChoice
		wantJSON    string
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name:     "mode",
			value:    hftypes.ChatToolChoice{Mode: &mode},
			wantJSON: `"provider-mode"`,
		},
		{
			name:     "null",
			value:    hftypes.ChatToolChoice{},
			wantJSON: `null`,
		},
		{
			name:     "function",
			value:    hftypes.ChatToolChoice{Function: &hftypes.ChatFunctionName{Name: "do"}},
			wantJSON: `{"function":{"name":"do"}}`,
		},
		{
			name: "both",
			value: hftypes.ChatToolChoice{
				Mode:     &mode,
				Function: &hftypes.ChatFunctionName{Name: "do"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "empty mode",
			value:       hftypes.ChatToolChoice{Mode: new(hftypes.ToolChoiceMode(""))},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
			if tc.wantJSON != "" && string(data) != tc.wantJSON {
				t.Fatalf("unexpected json: %s", string(data))
			}
		})
	}
}

func TestChatResponseFormat(t *testing.T) {
	t.Parallel()

	providerType := hftypes.ResponseFormatType("provider-format")

	cases := []struct {
		name        string
		value       hftypes.ChatResponseFormat
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name: "json_schema",
			value: hftypes.ChatResponseFormat{
				Type:       hftypes.ResponseFormatTypeJSONSchema,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "n"},
			},
		},
		{
			name:  "provider type",
			value: hftypes.ChatResponseFormat{Type: providerType},
		},
		{
			name:  "text",
			value: hftypes.ChatResponseFormat{Type: hftypes.ResponseFormatTypeText},
		},
		{
			name:  "json_object",
			value: hftypes.ChatResponseFormat{Type: hftypes.ResponseFormatTypeJSONObject},
		},
		{
			name:        "json_schema missing",
			value:       hftypes.ChatResponseFormat{Type: hftypes.ResponseFormatTypeJSONSchema},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "other with json_schema",
			value: hftypes.ChatResponseFormat{
				Type:       providerType,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "n"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "text with json_schema",
			value: hftypes.ChatResponseFormat{
				Type:       hftypes.ResponseFormatTypeText,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "n"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "json_object with json_schema",
			value: hftypes.ChatResponseFormat{
				Type:       hftypes.ResponseFormatTypeJSONObject,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: "n"},
			},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name:        "empty type",
			value:       hftypes.ChatResponseFormat{},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "json_schema empty name",
			value: hftypes.ChatResponseFormat{
				Type:       hftypes.ResponseFormatTypeJSONSchema,
				JSONSchema: &hftypes.ChatJSONSchemaConfig{Name: ""},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatFunctionDefinition_Validation(t *testing.T) {
	t.Parallel()

	_, err := json.Marshal(hftypes.ChatFunctionDefinition{})
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestChatFunctionName_MarshalValidation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		value       hftypes.ChatFunctionName
		wantJSON    string
		wantErr     bool
		wantErrKind hferrors.SDKErrorKind
	}{
		{
			name:     "success",
			value:    hftypes.ChatFunctionName{Name: "fn"},
			wantJSON: `{"name":"fn"}`,
		},
		{
			name:        "missing name",
			value:       hftypes.ChatFunctionName{},
			wantErr:     true,
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.value)
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, tc.wantErrKind)

				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantJSON, string(data))
		})
	}
}

func TestChatTool_MarshalTypeMissing(t *testing.T) {
	t.Parallel()

	value := hftypes.ChatTool{
		Function: hftypes.ChatFunctionDefinition{Name: "fn"},
	}
	_, err := json.Marshal(value)
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
}

func TestChatToolCall_MarshalValidation(t *testing.T) {
	t.Parallel()

	cases := []toolCallMarshalCase{
		{
			name: "missing type",
			value: hftypes.ChatToolCall{
				ID:       "id",
				Function: hftypes.ChatFunctionCall{Name: "fn", Arguments: "{}"},
			},
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "missing id",
			value: hftypes.ChatToolCall{
				Type:     "function",
				Function: hftypes.ChatFunctionCall{Name: "fn", Arguments: "{}"},
			},
			wantErrKind: hferrors.SDKErrorKindConfiguration,
		},
		{
			name: "missing function name",
			value: hftypes.ChatToolCall{
				ID:       "id",
				Type:     "function",
				Function: hftypes.ChatFunctionCall{Arguments: "{}"},
			},
			wantErrKind: hferrors.SDKErrorKindValidation,
		},
		{
			name: "missing function arguments",
			value: hftypes.ChatToolCall{
				ID:       "id",
				Type:     "function",
				Function: hftypes.ChatFunctionCall{Name: "fn"},
			},
			wantErrKind: hferrors.SDKErrorKindValidation,
		},
	}

	runToolCallMarshalTests(t, cases)
}
