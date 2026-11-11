//go:build !integration

package request

import (
	"testing"
)

func FuzzParseSSEField(f *testing.F) {
	f.Add("data: hello")
	f.Add("")
	f.Add("data: ")
	f.Add("[DONE]")
	f.Add("data: {\"text\":\"hello\"}")
	f.Add("data: {\"text\":\"hello\"}\n\ndata: [DONE]\n\n")
	f.Add("invalid line without colon")
	f.Fuzz(func(_ *testing.T, line string) {
		_, _ = parseSSEField(line)
	})
}
