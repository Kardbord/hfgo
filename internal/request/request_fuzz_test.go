//go:build !integration

package request

import (
	"bytes"
	"testing"

	"github.com/Kardbord/hfgo/v4/hfopts"
)

func FuzzJoinURL(f *testing.F) {
	f.Add("https://example.com", "/api")
	f.Add("", "")
	f.Add("https://example.com", "")
	f.Add("", "/api")
	f.Add("https://example.com/api", "/../etc/passwd")
	f.Add("https://example.com/api", "/path?query=1")
	f.Add("https://example.com/api", "/path#fragment")
	f.Fuzz(func(_ *testing.T, baseURL, path string) {
		_, _ = joinURL(baseURL, path)
	})
}

func FuzzBuildHTTPRequest(f *testing.F) {
	f.Add("GET", "/test", []byte("hello"))
	f.Add("", "/test", []byte("hello"))
	f.Add("GET\n", "/test", []byte("hello"))
	f.Add("GET", "", []byte("hello"))
	f.Add("GET", "/test", []byte(""))
	f.Fuzz(func(_ *testing.T, method, path string, body []byte) {
		opts := hfopts.NewOptions()
		_, _ = buildHTTPRequest(opts, method, path, bytes.NewReader(body))
	})
}
