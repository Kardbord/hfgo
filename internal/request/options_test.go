//go:build !integration

package request

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/providers"
	"github.com/Kardbord/hfgo/v4/internal/sdkversion"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
)

func TestOptions_With(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		initial  Options
		options  []Option
		validate func(t *testing.T, orig, updated Options)
	}{
		{
			name:    "is immutable",
			initial: NewOptions(),
			options: []Option{
				WithToken("secret"),
			},
			validate: func(t *testing.T, orig, updated Options) {
				t.Helper()
				if orig.Token != "" {
					t.Errorf("expected original Token to be empty, got %q", orig.Token)
				}
				if updated.Token != "secret" {
					t.Errorf("expected updated Token to be 'secret', got %q", updated.Token)
				}
			},
		},
		{
			name:    "duplicate options - last wins",
			initial: NewOptions(),
			options: []Option{
				WithToken("first"),
				WithToken("second"),
			},
			validate: func(t *testing.T, _ Options, updated Options) {
				t.Helper()
				if updated.Token != "second" {
					t.Errorf("expected last option to win, got %q", updated.Token)
				}
			},
		},
		{
			name:    "multiple fields",
			initial: NewOptions(),
			options: []Option{
				WithToken("token123"),
				WithModel("llama-3"),
				WithProvider(providers.HuggingFaceProvider{}),
				WithUserAgent("myapp/1.2.3"),
			},
			validate: func(t *testing.T, _ Options, updated Options) {
				t.Helper()
				if updated.Token != "token123" {
					t.Errorf("expected Token 'token123', got %q", updated.Token)
				}
				if updated.Model != "llama-3" {
					t.Errorf("expected Model 'llama-3', got %q", updated.Model)
				}
				defaultProvider := providers.HuggingFaceProvider{}
				if updated.Provider != defaultProvider {
					t.Errorf("expected Provider %v, got %v", defaultProvider, updated.Provider)
				}
				if updated.UserAgent != "myapp/1.2.3" {
					t.Errorf("expected UserAgent 'myapp/1.2.3, got %q", updated.UserAgent)
				}
			},
		},
		{
			name:    "user agent suffix uses default when base is empty",
			initial: NewOptions(),
			options: []Option{
				WithUserAgent(""),
				WithUserAgentSuffix("custom/1.0"),
			},
			validate: func(t *testing.T, _ Options, updated Options) {
				t.Helper()
				want := sdkversion.UserAgent() + " custom/1.0"
				if updated.UserAgent != want {
					t.Errorf("expected UserAgent %q, got %q", want, updated.UserAgent)
				}
			},
		},
		{
			name:    "user agent suffix uses existing base",
			initial: NewOptions(),
			options: []Option{
				WithUserAgent("myapp/2.0"),
				WithUserAgentSuffix("custom/1.0"),
			},
			validate: func(t *testing.T, _ Options, updated Options) {
				t.Helper()
				want := "myapp/2.0 custom/1.0"
				if updated.UserAgent != want {
					t.Errorf("expected UserAgent %q, got %q", want, updated.UserAgent)
				}
			},
		},
		{
			name:    "http client factory replaces default client",
			initial: NewOptions(),
			options: []Option{
				WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(
						testutils.NewMockTransport(http.StatusOK, `{}`, nil),
					)
				}),
			},
			validate: func(t *testing.T, orig, updated Options) {
				t.Helper()
				if orig.HTTPClient == nil || updated.HTTPClient == nil {
					t.Fatal("expected http clients to be set")
				}
				if orig.HTTPClient == updated.HTTPClient {
					t.Fatal("expected http client to be replaced")
				}
				if _, ok := updated.HTTPClient.Transport.(*testutils.MockTransport); !ok {
					t.Fatal("expected updated http client to use mock transport")
				}
			},
		},
		{
			name: "default http client resets custom client",
			initial: NewOptions().WithHTTPClientFactory(func() http.Client {
				return testutils.NewMockHTTPClient(
					testutils.NewMockTransport(http.StatusOK, `{}`, nil),
				)
			}),
			options: []Option{
				WithDefaultHTTPClient(),
			},
			validate: func(t *testing.T, orig, updated Options) {
				t.Helper()
				if orig.HTTPClient == nil || updated.HTTPClient == nil {
					t.Fatal("expected http clients to be set")
				}
				if _, ok := orig.HTTPClient.Transport.(*testutils.MockTransport); !ok {
					t.Fatal("expected original http client to use mock transport")
				}
				if _, ok := updated.HTTPClient.Transport.(*testutils.MockTransport); ok {
					t.Fatal("expected default http client to replace mock transport")
				}
			},
		},
	}

	for i := range tests {
		tt := tests[i]
		t.Run(tt.name, func(t *testing.T) {
			orig := tt.initial
			updated := orig.With(tt.options...)

			if tt.validate != nil {
				tt.validate(t, orig, updated)
			}
		})
	}
}

func TestOptions_WithHelpers(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

	opts := NewOptions().
		WithBaseURL("https://example.com").
		WithToken("token").
		WithModel("model").
		WithProvider(providers.HuggingFaceProvider{}).
		WithContext(ctx).
		WithDefaultHTTPClient().
		WithMaxResponseBodyBytes(42).
		WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		})

	if opts.BaseURL != "https://example.com" {
		t.Errorf("expected BaseURL to be set, got %q", opts.BaseURL)
	}
	if opts.Token != "token" {
		t.Errorf("expected Token to be set, got %q", opts.Token)
	}
	if opts.Model != "model" {
		t.Errorf("expected Model to be set, got %q", opts.Model)
	}
	defaultProvider := providers.HuggingFaceProvider{}
	if opts.Provider != defaultProvider {
		t.Errorf("expected Provider %v, got %v", defaultProvider, opts.Provider)
	}
	if opts.Context() != ctx {
		t.Error("expected context to be set")
	}
	if opts.MaxResponseBodyBytes != 42 {
		t.Errorf("expected MaxResponseBodyBytes to be 42, got %d", opts.MaxResponseBodyBytes)
	}
	if opts.HTTPClient == nil || opts.HTTPClient.Transport != mt {
		t.Error("expected HTTP client transport to be set")
	}
}

func TestWithUserAgentSuffix(t *testing.T) {
	t.Parallel()

	opts := NewOptions().WithUserAgentSuffix("custom/1.0")
	want := sdkversion.UserAgent() + " custom/1.0"
	if opts.UserAgent != want {
		t.Errorf("expected UserAgent %q, got %q", want, opts.UserAgent)
	}
}

func TestWithUserAgentSuffix_Empty(t *testing.T) {
	t.Parallel()

	t.Run("default user agent unchanged", func(t *testing.T) {
		opts := NewOptions().WithUserAgentSuffix("")
		want := sdkversion.UserAgent()
		if opts.UserAgent != want {
			t.Errorf("expected UserAgent %q, got %q", want, opts.UserAgent)
		}
	})

	t.Run("empty base remains empty", func(t *testing.T) {
		opts := NewOptions().WithUserAgent("").WithUserAgentSuffix("")
		if opts.UserAgent != "" {
			t.Errorf("expected empty UserAgent, got %q", opts.UserAgent)
		}
	})
}

func TestWithHeaders_CopiesMap(t *testing.T) {
	t.Parallel()

	headers := http.Header{"X-Test": []string{"one"}}
	opts := NewOptions().WithHeaders(headers)
	headers.Set("X-Test", "two")

	if opts.Headers == nil || opts.Headers.Get("X-Test") != "one" {
		t.Errorf("expected headers to be copied, got %#v", opts.Headers)
	}
}

func TestOptions_DefensiveHeaderClone(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

	tests := []struct {
		name  string
		apply func(Options) Options
	}{
		{
			name: "With",
			apply: func(opts Options) Options {
				return opts.With(WithToken("token"))
			},
		},
		{
			name: "WithBaseURL",
			apply: func(opts Options) Options {
				return opts.WithBaseURL("https://example.com")
			},
		},
		{
			name: "WithToken",
			apply: func(opts Options) Options {
				return opts.WithToken("token")
			},
		},
		{
			name: "WithModel",
			apply: func(opts Options) Options {
				return opts.WithModel("model")
			},
		},
		{
			name: "WithProvider",
			apply: func(opts Options) Options {
				return opts.WithProvider(providers.HuggingFaceProvider{})
			},
		},
		{
			name: "WithUserAgent",
			apply: func(opts Options) Options {
				return opts.WithUserAgent("ua/1.0")
			},
		},
		{
			name: "WithUserAgentSuffix",
			apply: func(opts Options) Options {
				return opts.WithUserAgentSuffix("custom/1.0")
			},
		},
		{
			name: "WithMaxResponseBodyBytes",
			apply: func(opts Options) Options {
				return opts.WithMaxResponseBodyBytes(42)
			},
		},
		{
			name: "WithContext",
			apply: func(opts Options) Options {
				return opts.WithContext(context.Background())
			},
		},
		{
			name: "WithHTTPClientFactory",
			apply: func(opts Options) Options {
				return opts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				})
			},
		},
		{
			name: "WithHeaders",
			apply: func(opts Options) Options {
				return opts.WithHeaders(http.Header{"X-Other": []string{"value"}})
			},
		},
		{
			name: "WithHeader",
			apply: func(opts Options) Options {
				return opts.WithHeader("X-Other", "value")
			},
		},
		{
			name: "WithDefaultHeader",
			apply: func(opts Options) Options {
				return opts.WithDefaultHeader("X-Test", "default")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			orig := NewOptions().WithHeader("X-Test", "one")
			derived := tt.apply(orig)

			orig.Headers.Set("X-Test", "two")
			if got := derived.Headers.Get("X-Test"); got != "one" {
				t.Errorf("expected derived header to stay 'one', got %q", got)
			}

			derived.Headers.Set("X-Test", "three")
			if got := orig.Headers.Get("X-Test"); got != "two" {
				t.Errorf("expected original header to stay 'two', got %q", got)
			}
		})
	}
}

func TestWithHeaders_CanonicalizesAndOverrides(t *testing.T) {
	t.Parallel()

	opts := NewOptions().
		WithHeaders(http.Header{"x-test": []string{"one"}}).
		WithHeader("X-TEST", "two")

	if got := opts.Headers.Get("X-Test"); got != "two" {
		t.Errorf("expected X-Test to be overridden to 'two', got %q", got)
	}
	for key := range opts.Headers {
		if key == "x-test" || key == "X-TEST" {
			t.Error("expected header keys to be canonicalized")
		}
	}
}

func TestWithDefaultHeader_CaseInsensitiveAndEmpty(t *testing.T) {
	t.Parallel()

	opts := NewOptions().
		WithHeader("x-test", "").
		WithDefaultHeader("X-Test", "default")

	if got := opts.Headers.Get("X-Test"); got != "default" {
		t.Errorf("expected default header value, got %q", got)
	}

	unchanged := opts.WithDefaultHeader("x-test", "other")
	if got := unchanged.Headers.Get("X-Test"); got != "default" {
		t.Errorf("expected default header to remain, got %q", got)
	}
}

func TestNewOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		validate func(t *testing.T, opts Options)
	}{
		{
			name: "has default context",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.Context() == nil {
					t.Error("expected default context, got nil")
				}
			},
		},
		{
			name: "has default BaseURL",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.BaseURL != DefaultBaseURL {
					t.Errorf("expected BaseURL %q, got %q", DefaultBaseURL, opts.BaseURL)
				}
			},
		},
		{
			name: "has default Token",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.Token != DefaultToken {
					t.Errorf("expected Token %q, got %q", DefaultToken, opts.Token)
				}
			},
		},
		{
			name: "has default Model",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.Model != DefaultModel {
					t.Errorf("expected Model %q, got %q", DefaultModel, opts.Model)
				}
			},
		},
		{
			name: "has default Provider",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.Provider == nil {
					t.Errorf("expected Provider not nil, got nil")
				}
			},
		},
		{
			name: "has default UserAgent",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.UserAgent != sdkversion.UserAgent() {
					t.Errorf(
						"expected UserAgent %q, got %q",
						sdkversion.UserAgent(),
						opts.UserAgent,
					)
				}
			},
		},
		{
			name: "has default headers",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.Headers != nil {
					t.Errorf("expected default headers to be nil, got %#v", opts.Headers)
				}
			},
		},
		{
			name: "has default HTTP client",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.HTTPClient == nil {
					t.Error("expected default http client, got nil")
				}
			},
		},
		{
			name: "has default max response body size",
			validate: func(t *testing.T, opts Options) {
				t.Helper()
				if opts.MaxResponseBodyBytes != DefaultMaxResponseBodyBytes {
					t.Errorf(
						"expected %d max response body bytes, got %d",
						DefaultMaxResponseBodyBytes,
						opts.MaxResponseBodyBytes,
					)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := NewOptions()
			if tt.validate != nil {
				tt.validate(t, opts)
			}
		})
	}
}

func TestOptions_Validate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		opts    Options
		wantErr bool
		kind    hferrors.SDKErrorKind
	}{
		{
			name:    "valid options",
			opts:    NewOptions(),
			wantErr: false,
		},
		{
			name:    "nil http client",
			opts:    NewOptions().WithHTTPClientFactory(nil),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
		{
			name:    "invalid base URL",
			opts:    NewOptions().WithBaseURL("http://[::1"),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
		{
			name:    "base URL missing scheme",
			opts:    NewOptions().WithBaseURL("example.com/api"),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
		{
			name:    "base URL missing host",
			opts:    NewOptions().WithBaseURL("https:///api"),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
		{
			name:    "base URL with query",
			opts:    NewOptions().WithBaseURL("https://example.com/api?token=abc"),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
		{
			name:    "base URL with fragment",
			opts:    NewOptions().WithBaseURL("https://example.com/api#section"),
			wantErr: true,
			kind:    hferrors.SDKErrorKindConfiguration,
		},
	}

	for i := range tests {
		tt := tests[i]
		t.Run(tt.name, func(t *testing.T) {
			err := tt.opts.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var sdkErr *hferrors.SDKError
				if !errors.As(err, &sdkErr) {
					t.Fatalf("expected SDKError, got %T", err)
				}
				if sdkErr.Kind != tt.kind {
					t.Fatalf("expected SDKError kind %q, got %q", tt.kind, sdkErr.Kind)
				}
			}
		})
	}
}
