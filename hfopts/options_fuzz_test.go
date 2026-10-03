//go:build !integration

package hfopts_test

import (
	"testing"

	"github.com/Kardbord/hfgo/v4/hfopts"
)

func FuzzOptionsValidate(f *testing.F) {
	f.Add("https://example.com")
	f.Add("")
	f.Add("example.com")
	f.Add("https://example.com/api?token=secret")
	f.Add("https://example.com/api#section")
	f.Add("https://example.com/api?token=secret#section")
	f.Add("http://[::1")
	f.Fuzz(func(_ *testing.T, baseURL string) {
		opts := hfopts.NewOptions().With(hfopts.WithBaseURL(baseURL))
		_ = opts.Validate()
	})
}
