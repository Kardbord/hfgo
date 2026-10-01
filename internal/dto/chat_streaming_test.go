//go:build !integration

package dto

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestChatStreamDelta_UnmarshalValidation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		unmarshal string
		wantErr   bool
	}{
		{
			name:      "tool_calls with content",
			unmarshal: `{"content":"hi","tool_calls":[{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}]}`,
			wantErr:   true,
		},
		{
			name:      "tool_calls with tool_call_id",
			unmarshal: `{"tool_call_id":"id","tool_calls":[{"id":"id","type":"function","index":0,"function":{"name":"fn","arguments":"{}"}}]}`,
			wantErr:   true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatStreamDelta
			err := json.Unmarshal([]byte(tc.unmarshal), &got)
			if tc.wantErr {
				require.Error(t, err)

				return
			}
			require.NoError(t, err)
		})
	}
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
			wantRole:  testutils.Ptr("assistant"),
		},
		{
			name:        "content only",
			unmarshal:   `{"content":"hi"}`,
			wantContent: testutils.Ptr("hi"),
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
