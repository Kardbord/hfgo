//go:build !integration

package hftypes_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// FuzzChatResponseValidate round-trips arbitrary JSON through
// [hftypes.ChatResponse] and its opt-in Validate. Decoding must be tolerant
// (never panic), Validate must return nil or a validation SDKError, and a
// successfully decoded response must re-marshal.
func FuzzChatResponseValidate(f *testing.F) {
	f.Add(
		[]byte(
			`{"id":"1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`,
		),
	)
	f.Add(
		[]byte(
			`{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
		),
	)
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"choices":[]}`))
	f.Add([]byte(`{"choices":[{"message":{}}]}`))
	f.Add(
		[]byte(
			`{"choices":[{"message":{"role":"assistant","content":"hi","tool_calls":[{"id":"1","type":"function","function":{"name":"f"}}]}}]}`,
		),
	)
	f.Add(
		[]byte(
			`{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"1","type":"function"}]}}]}`,
		),
	)
	f.Add([]byte(`{"choices":[{"message":null}]}`))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))
	f.Add(
		[]byte(
			`{"choices":[{"logprobs":{"content":[{"token":"a","logprob":-0.5,"top_logprobs":[{"token":"a","logprob":-0.5}]}]}}]}`,
		),
	)
	f.Add([]byte(`{"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		var resp hftypes.ChatResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			return
		}

		if err := resp.Validate(); err != nil {
			t.Logf("Validate returned (expected for schema-violating bodies): %v", err)
		}

		assertRemarshal(t, resp)
	})
}

// FuzzChatStreamResponseValidate round-trips arbitrary JSON through
// [hftypes.ChatStreamResponse] and its opt-in Validate, covering the
// streaming delta union (text content vs tool calls).
func FuzzChatStreamResponseValidate(f *testing.F) {
	f.Add([]byte(`{"id":"1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`))
	f.Add([]byte(`{"choices":[{"delta":{"role":"assistant"}}]}`))
	f.Add(
		[]byte(
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"1","type":"function","function":{"name":"f","arguments":""}}]}}]}`,
		),
	)
	f.Add(
		[]byte(
			`{"choices":[{"delta":{"content":"hi","tool_calls":[{"id":"1","type":"function","function":{"name":"f"}}]}}]}`,
		),
	)
	f.Add([]byte(`{"choices":[{"delta":{"tool_call_id":"1","content":"mixed"}}]}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"choices":[]}`))
	f.Add([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
	f.Add([]byte(`{"usage":{"prompt_tokens":1}}`))
	f.Add([]byte(`null`))
	f.Add([]byte(`[]`))

	f.Fuzz(func(t *testing.T, data []byte) {
		var resp hftypes.ChatStreamResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			return
		}

		if err := resp.Validate(); err != nil {
			t.Logf("Validate returned (expected for schema-violating bodies): %v", err)
		}

		assertRemarshal(t, resp)
	})
}

// assertRemarshal re-marshals a decoded value and checks the documented
// asymmetry between tolerant decoding and strict marshaling: a marshal
// failure on a schema-violating value is allowed but must be a well-typed
// *hferrors.SDKError (the strict MarshalJSON implementations raise
// validation errors, never bare errors), and a successful marshal is
// idempotent (marshaling the re-decoded value reproduces the same bytes).
func assertRemarshal(t *testing.T, resp any) {
	t.Helper()

	out, err := json.Marshal(resp)
	if err != nil {
		if _, ok := errors.AsType[*hferrors.SDKError](err); !ok {
			t.Errorf("re-marshal failed with unexpected error type: %v (%T)", err, err)
		}

		return
	}

	// Re-decode into the same concrete type and re-marshal: the marshal
	// output must be idempotent (same bytes), i.e. marshal∘unmarshal is a
	// fixed point. Decoding into a generic map would compare struct field
	// order against map key ordering, so reflect is required here.
	again := reflect.New(reflect.TypeOf(resp))
	if err := json.Unmarshal(out, again.Interface()); err != nil {
		t.Errorf("marshaled output cannot be re-decoded into %T: %v", resp, err)

		return
	}

	out2, err := json.Marshal(again.Interface())
	if err != nil {
		t.Errorf("re-decoded marshaled output cannot re-marshal: %v", err)

		return
	}
	if !bytes.Equal(out, out2) {
		t.Errorf("marshal is not idempotent: %s then %s", out, out2)
	}
}
