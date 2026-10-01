//go:build !integration

package dto_test

import (
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/internal/dto"
)

func FuzzChatRequestValidate(f *testing.F) {
	f.Add([]byte(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`))
	f.Add([]byte(``))
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":"test"}`))
	f.Add([]byte(`{"messages":[{"role":"user","content":"hi"}]}`))
	f.Fuzz(func(_ *testing.T, data []byte) {
		var req dto.ChatRequest
		if err := json.Unmarshal(data, &req); err != nil {
			return
		}
		_, err := json.Marshal(req)
		if err != nil {
			return
		}
	})
}

func FuzzChatMessageValidate(f *testing.F) {
	f.Add([]byte(`{"role":"user","content":"hi"}`))
	f.Add([]byte(``))
	f.Add([]byte(`{}`))
	f.Add([]byte(
		`{"role":"user","content":"hi","tool_calls":[` +
			`{"id":"1","type":"function","function":{"name":"test"}}]}`))
	f.Fuzz(func(_ *testing.T, data []byte) {
		var msg hfgo.ChatMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return
		}
		_, err := json.Marshal(msg)
		if err != nil {
			return
		}
	})
}
