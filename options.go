package hfgo

import (
	"context"
	"net/http"

	"github.com/Kardbord/hfgo/v4/internal/request"
)

// WithBaseURL returns an Option that sets the base URL for API requests.
// The base URL is the root endpoint for all HuggingFace API calls and must not
// include query parameters or fragments.
func WithBaseURL(u string) Option {
	return request.WithBaseURL(u)
}

// WithToken returns an Option that sets the authentication token for API requests.
// The token is used for Bearer authentication with the HuggingFace API.
func WithToken(t string) Option {
	return request.WithToken(t)
}

// WithModel returns an Option that sets the model to use for API requests.
// The model specifies which HuggingFace model should process the request.
func WithModel(m string) Option {
	return request.WithModel(m)
}

// WithProvider returns an Option that sets the provider for API requests.
// The provider specifies which inference provider should handle the request.
func WithProvider(p Provider) Option {
	return request.WithProvider(p)
}

// WithDefaultProvider returns an Option that sets the provider to the default
// HuggingFace provider.
func WithDefaultProvider() Option {
	return request.WithDefaultProvider()
}

// WithHTTPClientFactory returns an Option that sets a http.Client created by the factory.
// The factory is invoked when request options are applied, so it can be used per request
// or at client construction time.
// The factory should return a fresh client value; avoid sharing mutable internals like Transport unless synchronized.
// If the factory is nil, the HTTP client is set to nil.
func WithHTTPClientFactory(factory func() http.Client) Option {
	return request.WithHTTPClientFactory(factory)
}

// WithContext returns an Option that sets the context for API requests.
// The context can be used for cancellation, timeouts, and passing request-scoped values.
// If a nil context is provided, the SDK will fall back to context.Background().
func WithContext(ctx context.Context) Option {
	return request.WithContext(ctx)
}

// WithDefaultHTTPClient returns an Option that sets the default HTTP client.
func WithDefaultHTTPClient() Option {
	return request.WithDefaultHTTPClient()
}

// WithUserAgentSuffix returns an Option that appends a suffix to the SDK user agent string.
func WithUserAgentSuffix(s string) Option {
	return request.WithUserAgentSuffix(s)
}

// WithMaxResponseBodyBytes returns an Option that sets the maximum number of bytes
// read from any response body. Values <= 0 fall back to the default.
func WithMaxResponseBodyBytes(n int64) Option {
	return request.WithMaxResponseBodyBytes(n)
}

// WithHeaders returns an Option that sets custom headers applied to every request,
// overriding any existing values for matching keys.
// Per-request headers can still override these values when provided.
func WithHeaders(h http.Header) Option {
	return request.WithHeaders(h)
}

// WithHeader returns an Option that sets a single header applied to every request.
func WithHeader(key, value string) Option {
	return request.WithHeader(key, value)
}

// WithDefaultHeader returns an Option that sets a header only if missing or empty.
func WithDefaultHeader(key, value string) Option {
	return request.WithDefaultHeader(key, value)
}
