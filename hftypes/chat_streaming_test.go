//go:build !integration

package hftypes

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/stretchr/testify/require"
)

func TestChatStreamDelta_UnmarshalTolerant(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		unmarshal string
	}{
		{
			name:      "tool_calls with content",
			unmarshal: `{"content":"hi","tool_calls":[{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}]}`,
		},
		{
			name:      "tool_calls with tool_call_id",
			unmarshal: `{"tool_call_id":"id","tool_calls":[{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatStreamDelta
			require.NoError(t, json.Unmarshal([]byte(tc.unmarshal), &got))
		})
	}
}

func TestChatStreamDelta_Validate(t *testing.T) {
	t.Parallel()

	toolCalls := []ChatStreamToolCall{{
		ID:    "id",
		Type:  "function",
		Index: 0,
		Function: ChatStreamFunction{
			Name:      "fn",
			Arguments: "{}",
		},
	}}

	cases := []struct {
		name    string
		delta   ChatStreamDelta
		wantErr bool
	}{
		{
			name: "tool_calls with content",
			delta: ChatStreamDelta{
				Content:   new("hi"),
				ToolCalls: toolCalls,
			},
			wantErr: true,
		},
		{
			name: "tool_calls with tool_call_id",
			delta: ChatStreamDelta{
				ToolCallID: new("id"),
				ToolCalls:  toolCalls,
			},
			wantErr: true,
		},
		{
			name:  "text delta",
			delta: ChatStreamDelta{Content: new("hi")},
		},
		{
			name:  "tool call delta",
			delta: ChatStreamDelta{ToolCalls: toolCalls},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.delta.Validate()
			if tc.wantErr {
				require.Error(t, err)
				var sdkErr *hferrors.SDKError
				require.ErrorAs(t, err, &sdkErr)
				require.Equal(t, hferrors.SDKErrorKindValidation, sdkErr.Kind)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatStreamResponse_Validate(t *testing.T) {
	t.Parallel()

	require.NoError(t, ChatStreamResponse{}.Validate())

	valid := ChatStreamResponse{
		Choices: []ChatStreamChoice{{
			Delta: ChatStreamDelta{Content: new("hi")},
		}},
	}
	require.NoError(t, valid.Validate())

	invalid := ChatStreamResponse{
		Choices: []ChatStreamChoice{
			{Delta: ChatStreamDelta{Content: new("hi")}},
			{
				Delta: ChatStreamDelta{
					Content: new("hi"),
					ToolCalls: []ChatStreamToolCall{{
						Index: 0,
					}},
				},
			},
		},
	}
	err := invalid.Validate()
	require.Error(t, err)
	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindValidation, sdkErr.Kind)
	require.ErrorContains(t, err, "choice 1")
}

func TestChatStreamDelta_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		unmarshal     string
		wantRole      *string
		wantContent   *string
		wantToolCalls int
	}{
		{
			name:      "role only",
			unmarshal: `{"role":"assistant"}`,
			wantRole:  new("assistant"),
		},
		{
			name:        "content only",
			unmarshal:   `{"content":"hi"}`,
			wantContent: new("hi"),
		},
		{
			name:          "tool_calls only",
			unmarshal:     `{"tool_calls":[{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}]}`,
			wantToolCalls: 1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatStreamDelta
			require.NoError(t, json.Unmarshal([]byte(tc.unmarshal), &got))
			if tc.wantRole != nil {
				if got.Role == nil || *got.Role != *tc.wantRole {
					t.Fatalf("unexpected role: %+v", got.Role)
				}
				require.Nil(t, got.Content)
				require.Nil(t, got.ToolCallID)
				require.Nil(t, got.ToolCalls)
			}
			if tc.wantContent != nil {
				if got.Content == nil || *got.Content != *tc.wantContent {
					t.Fatalf("unexpected content: %+v", got.Content)
				}
				require.Nil(t, got.Role)
				require.Nil(t, got.ToolCallID)
				require.Nil(t, got.ToolCalls)
			}
			if tc.wantToolCalls > 0 {
				if len(got.ToolCalls) != tc.wantToolCalls {
					t.Fatalf("unexpected tool calls: %+v", got.ToolCalls)
				}
				call := got.ToolCalls[0]
				if call.ID != "id" || call.Type != "function" || call.Index != 0 ||
					call.Function.Name != "fn" ||
					call.Function.Arguments != "{}" {
					t.Fatalf("unexpected tool call: %+v", call)
				}
				require.Nil(t, got.Content)
				require.Nil(t, got.ToolCallID)
			}
		})
	}
}

func TestChatStreamToolCall_UnmarshalPartial(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		data []byte
	}{
		{
			name: "type missing",
			data: []byte(`{"id":"id","index":0,"function":{"name":"fn","arguments":"{}"}}`),
		},
		{
			name: "id missing",
			data: []byte(
				`{"type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}`,
			),
		},
		{
			name: "function name missing",
			data: []byte(
				`{"id":"id","type":"function","index":0,"function":{"arguments":"{}"}}`,
			),
		},
		{
			name: "function arguments missing",
			data: []byte(
				`{"id":"id","type":"function","index":0,"function":{"name":"fn"}}`,
			),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out ChatStreamToolCall
			err := json.Unmarshal(tc.data, &out)
			require.NoError(t, err)
		})
	}
}

func TestChatStreamToolCall_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(
		`{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}`,
	)
	var got ChatStreamToolCall
	require.NoError(t, json.Unmarshal(data, &got))
	if got.ID != "id" || got.Type != "function" || got.Index != 0 || got.Function.Name != "fn" {
		t.Fatalf("unexpected value: %+v", got)
	}
}

func TestChatStreamResponse_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(
		`{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"delta":{"role":"assistant"},"index":0}]}`,
	)
	var got ChatStreamResponse
	require.NoError(t, json.Unmarshal(data, &got))
	if got.ID != "id" || len(got.Choices) != 1 {
		t.Fatalf("unexpected response: %+v", got)
	}
	if got.Choices[0].Delta.Role == nil || *got.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("unexpected delta role: %+v", got.Choices[0].Delta.Role)
	}
}

func TestChatStreamFunction_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(`{"name":"fn","arguments":"{}"}`)
	var got ChatStreamFunction
	require.NoError(t, json.Unmarshal(data, &got))
	if got.Name != "fn" || got.Arguments != "{}" {
		t.Fatalf("unexpected stream function: %+v", got)
	}
}

func TestChatStreamChoice_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(`{"delta":{"role":"assistant"},"index":0}`)
	var got ChatStreamChoice
	require.NoError(t, json.Unmarshal(data, &got))
	if got.Index != 0 || got.Delta.Role == nil || *got.Delta.Role != "assistant" {
		t.Fatalf("unexpected stream choice: %+v", got)
	}
}

func TestToolCallAccumulator_Merge(t *testing.T) {
	t.Parallel()

	acc := toolCallAccumulator{}

	// Initial update should store and return provided values.
	id, typ, name := acc.merge(0, 0, "call_1", "function", "fn")
	if id != "call_1" || typ != "function" || name != "fn" {
		t.Fatalf("unexpected first merge: %q %q %q", id, typ, name)
	}

	// Missing fields should fall back to cached state.
	id, typ, name = acc.merge(0, 0, "", "", "")
	if id != "call_1" || typ != "function" || name != "fn" {
		t.Fatalf("missing fields should reuse cache: %q %q %q", id, typ, name)
	}

	// Updating only one field should preserve others.
	id, typ, name = acc.merge(0, 0, "", "", "fn_override")
	if id != "call_1" || typ != "function" || name != "fn_override" {
		t.Fatalf("partial update failed: %q %q %q", id, typ, name)
	}

	// Separate choice/call indices should keep isolated caches.
	idB, typB, nameB := acc.merge(1, 2, "call_2", "function", "fn2")
	if idB != "call_2" || typB != "function" || nameB != "fn2" {
		t.Fatalf("unexpected second call: %q %q %q", idB, typB, nameB)
	}
	id, typ, name = acc.merge(0, 0, "", "", "")
	if name != "fn_override" {
		t.Fatalf("first call cache should remain untouched: %q %q %q", id, typ, name)
	}
}

func TestToolCallAccumulator_MergeWithoutMetadata(t *testing.T) {
	t.Parallel()

	acc := toolCallAccumulator{}

	id, typ, name := acc.merge(0, 0, "", "", "")
	if id != "" || typ != "" || name != "" {
		t.Fatalf("expected empty metadata, got %q %q %q", id, typ, name)
	}

	id, typ, name = acc.merge(0, 0, "call_1", "", "")
	if id != "call_1" || typ != "" || name != "" {
		t.Fatalf("unexpected selective metadata: %q %q %q", id, typ, name)
	}

	id, typ, name = acc.merge(0, 0, "", "", "")
	if id != "call_1" || typ != "" || name != "" {
		t.Fatalf("expected cached id only: %q %q %q", id, typ, name)
	}
}
