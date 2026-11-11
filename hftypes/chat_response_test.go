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

func TestChatCompletionMessage_UnmarshalTolerant(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		unmarshal string
	}{
		{
			name:      "both content and tool_calls",
			unmarshal: `{"role":"assistant","content":"hi","tool_calls":[{"id":"id","type":"function","function":{"name":"fn","arguments":"{}"}}]}`,
		},
		{
			name:      "neither content nor tool_calls",
			unmarshal: `{"role":"assistant"}`,
		},
		{
			name:      "tool_call_id without content",
			unmarshal: `{"role":"assistant","tool_call_id":"id"}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got hftypes.ChatCompletionMessage
			require.NoError(t, json.Unmarshal([]byte(tc.unmarshal), &got))
		})
	}
}

func TestChatCompletionMessage_Validate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		message hftypes.ChatCompletionMessage
		wantErr bool
	}{
		{
			name: "content and tool_calls",
			message: hftypes.ChatCompletionMessage{
				Content: new("hi"),
				ToolCalls: []hftypes.ChatToolCallOutput{{
					ID:   "id",
					Type: "function",
					Function: hftypes.ChatFunctionCall{
						Name:      "fn",
						Arguments: "{}",
					},
				}},
			},
			wantErr: true,
		},
		{
			name:    "neither content nor tool_calls",
			message: hftypes.ChatCompletionMessage{},
			wantErr: true,
		},
		{
			name: "tool_call_id without content",
			message: hftypes.ChatCompletionMessage{
				ToolCallID: new("id"),
			},
			wantErr: true,
		},
		{
			name: "tool call missing id",
			message: hftypes.ChatCompletionMessage{
				ToolCalls: []hftypes.ChatToolCallOutput{{
					Type: "function",
					Function: hftypes.ChatFunctionCall{
						Name:      "fn",
						Arguments: "{}",
					},
				}},
			},
			wantErr: true,
		},
		{
			name: "valid text message",
			message: hftypes.ChatCompletionMessage{
				Content: new("hi"),
			},
		},
		{
			name: "valid tool call message",
			message: hftypes.ChatCompletionMessage{
				ToolCalls: []hftypes.ChatToolCallOutput{{
					ID:   "id",
					Type: "function",
					Function: hftypes.ChatFunctionCall{
						Name:      "fn",
						Arguments: "{}",
					},
				}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.message.Validate()
			if tc.wantErr {
				require.Error(t, err)
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindValidation)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatResponse_Validate(t *testing.T) {
	t.Parallel()

	valid := hftypes.ChatResponse{
		Choices: []hftypes.ChatChoice{{
			Message: hftypes.ChatCompletionMessage{Content: new("hi")},
		}},
	}
	require.NoError(t, valid.Validate())

	empty := hftypes.ChatResponse{}
	err := empty.Validate()
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindValidation)
	require.ErrorContains(t, err, "no choices")

	invalid := hftypes.ChatResponse{
		Choices: []hftypes.ChatChoice{
			{Message: hftypes.ChatCompletionMessage{Content: new("hi")}},
			{},
		},
	}
	err = invalid.Validate()
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindValidation)
	require.ErrorContains(t, err, "choice 1")
}

func TestChatToolCallOutput_UnmarshalTolerant(t *testing.T) {
	t.Parallel()

	cases := []toolCallDecodeCase{
		{
			name: "type empty",
			data: []byte(`{"id":"id","type":"","function":{"name":"fn","arguments":"{}"}}`),
		},
		{
			name: "type missing",
			data: []byte(`{"id":"id","function":{"name":"fn","arguments":"{}"}}`),
		},
		{
			name: "id missing",
			data: []byte(
				`{"type":"function","function":{"name":"fn","arguments":"{}"}}`,
			),
		},
		{
			name: "function name missing",
			data: []byte(
				`{"id":"id","type":"function","function":{"arguments":"{}"}}`,
			),
		},
		{
			name: "function arguments missing",
			data: []byte(
				`{"id":"id","type":"function","function":{"name":"fn"}}`,
			),
		},
	}

	runToolCallDecodeTests(t, cases, func(data []byte) error {
		var out hftypes.ChatToolCallOutput

		return json.Unmarshal(data, &out)
	})
}

func TestChatToolCallOutput_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(`{"id":"id","type":"function","function":{"name":"fn","arguments":"{}"}}`)
	var got hftypes.ChatToolCallOutput
	require.NoError(t, json.Unmarshal(data, &got))
	if got.ID != "id" || got.Type != "function" || got.Function.Name != "fn" ||
		got.Function.Arguments != "{}" {
		t.Fatalf("unexpected value: %+v", got)
	}
}

func TestChatResponse_UnmarshalTypeValidation(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		unmarshal string
		wantErr   bool
	}{
		{
			name:      "usage total_tokens string",
			unmarshal: `{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":"3"}}`,
			wantErr:   true,
		},
		{
			name:      "choices not array",
			unmarshal: `{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":{},"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
			wantErr:   true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got hftypes.ChatResponse
			err := json.Unmarshal([]byte(tc.unmarshal), &got)
			if tc.wantErr {
				require.Error(t, err)

				return
			}
			require.NoError(t, err)
		})
	}
}

func TestChatResponse_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(
		`{"id":"id","created":1,"model":"m","system_fingerprint":"s","choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
	)
	var got hftypes.ChatResponse
	require.NoError(t, json.Unmarshal(data, &got))
	if got.ID != "id" || got.Model != "m" || len(got.Choices) != 1 {
		t.Fatalf("unexpected response: %+v", got)
	}
	if got.Choices[0].Message.Content == nil || *got.Choices[0].Message.Content != "hi" {
		t.Fatalf("unexpected message content: %+v", got.Choices[0].Message.Content)
	}
}

func TestChatLogProbs_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(
		`{"content":[{"token":"t","logprob":0.1,"top_logprobs":[{"token":"t","logprob":0.1}]}]}`,
	)
	var got hftypes.ChatLogProbs
	require.NoError(t, json.Unmarshal(data, &got))
	if len(got.Content) != 1 || got.Content[0].Token != "t" {
		t.Fatalf("unexpected logprobs: %+v", got)
	}
}

func TestChatUsage_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(`{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}`)
	var got hftypes.ChatUsage
	require.NoError(t, json.Unmarshal(data, &got))
	if got.PromptTokens != 1 || got.CompletionTokens != 2 || got.TotalTokens != 3 {
		t.Fatalf("unexpected usage: %+v", got)
	}
}

func TestChatFunctionCall_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(`{"name":"fn","arguments":"{}"}`)
	var got hftypes.ChatFunctionCall
	require.NoError(t, json.Unmarshal(data, &got))
	if got.Name != "fn" || got.Arguments != "{}" {
		t.Fatalf("unexpected function call: %+v", got)
	}
}

func TestChatChoice_UnmarshalSuccess(t *testing.T) {
	t.Parallel()

	data := []byte(
		`{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hi"}}`,
	)
	var got hftypes.ChatChoice
	require.NoError(t, json.Unmarshal(data, &got))
	if got.FinishReason != "stop" || got.Message.Content == nil || *got.Message.Content != "hi" {
		t.Fatalf("unexpected choice: %+v", got)
	}
}
