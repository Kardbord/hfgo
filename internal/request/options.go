package request

import (
	"context"
	"fmt"
	"net/http"
	"net/textproto"
	"net/url"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/providers"
	"github.com/Kardbord/hfgo/v4/internal/sdkversion"
)

// Options holds configuration settings for API requests.
// Built-in option helpers return a new value and defensively clone headers,
// while context and the HTTP client are shared as-is. Custom options should avoid
// reusing mutable header maps if they want the same defensive-copy behavior.
type Options struct {
	ctx                  context.Context //nolint:containedctx // Stored default context; per-request code normalizes it before use.
	BaseURL              string
	Token                string
	Model                string
	Provider             providers.Provider
	UserAgent            string
	Headers              http.Header
	MaxResponseBodyBytes int64
	HTTPClient           *http.Client
}

const (
	// DefaultBaseURL is the default HuggingFace API endpoint.
	DefaultBaseURL = "https://router.huggingface.co"
	// DefaultToken is the default authentication token (empty string).
	DefaultToken = ""
	// DefaultModel is the default model to use.
	DefaultModel = ""
	// DefaultMaxResponseBodyBytes caps the amount of response data read into memory by default.
	DefaultMaxResponseBodyBytes int64 = 1 << 20 // 1 MiB
)

// DefaultHTTPClient returns a new HTTP client configured with a cloned default transport.
func DefaultHTTPClient() *http.Client {
	defaultTransport, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return &http.Client{
			Transport:     nil,
			CheckRedirect: nil,
			Jar:           nil,
			Timeout:       0,
		}
	}

	return &http.Client{
		Transport:     defaultTransport.Clone(),
		CheckRedirect: nil,
		Jar:           nil,
		Timeout:       0,
	}
}

// NewOptions creates a new Options instance with default values.
// The returned options use a background context, default endpoints, and the default HTTP client.
func NewOptions() Options {
	return Options{
		ctx:                  context.Background(),
		BaseURL:              DefaultBaseURL,
		Token:                DefaultToken,
		Model:                DefaultModel,
		Provider:             providers.HuggingFaceProvider{},
		UserAgent:            sdkversion.UserAgent(),
		Headers:              nil,
		MaxResponseBodyBytes: DefaultMaxResponseBodyBytes,
		HTTPClient:           DefaultHTTPClient(),
	}
}

// Option is a function type that modifies Options.
// It follows the functional options pattern for flexible configuration.
// Custom options that set Headers should avoid reusing mutable header maps
// if they want to preserve the defensive-copy behavior of built-in helpers.
type Option func(*Options)

// Validate returns a configuration error if the options are invalid.
func (o Options) Validate() error {
	if o.HTTPClient == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "http client is nil",
			Err:     nil,
		}
	}
	parsedBase, err := url.Parse(o.BaseURL)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", o.BaseURL),
			Err:     err,
		}
	}
	if parsedBase.Scheme == "" || parsedBase.Host == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL must include scheme and host, got %q", o.BaseURL),
			Err:     nil,
		}
	}
	if parsedBase.RawQuery != "" || parsedBase.Fragment != "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL must not include query or fragment, got %q", o.BaseURL),
			Err:     nil,
		}
	}
	if o.Provider == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "provider must not be nil",
			Err:     nil,
		}
	}

	return nil
}

// With returns a new Options instance with the provided options applied.
// This method creates a copy of the current options and applies modifications to it.
func (o Options) With(opts ...Option) Options {
	o = o.clone()
	o.apply(opts...)

	return o
}

// WithBaseURL returns a new Options instance with the base URL updated.
// The base URL must not include query parameters or fragments.
func (o Options) WithBaseURL(u string) Options {
	o = o.clone()
	o.BaseURL = u

	return o
}

// WithToken returns a new Options instance with the authentication token updated.
func (o Options) WithToken(t string) Options {
	o = o.clone()
	o.Token = t

	return o
}

// WithModel returns a new Options instance with the model updated.
func (o Options) WithModel(m string) Options {
	o = o.clone()
	o.Model = m

	return o
}

// WithProvider returns a new Options instance with the provider updated.
func (o Options) WithProvider(p providers.Provider) Options {
	o = o.clone()
	o.Provider = p

	return o
}

// WithDefaultProvider returns a new Options instance with the default
// HuggingFace provider.
func (o Options) WithDefaultProvider() Options {
	o = o.clone()
	o.Provider = providers.HuggingFaceProvider{}

	return o
}

// WithUserAgent returns a new Options instance with the User-Agent value updated.
func (o Options) WithUserAgent(ua string) Options {
	o = o.clone()
	o.UserAgent = ua

	return o
}

// WithUserAgentSuffix returns a new Options instance with a suffix appended to the SDK User-Agent.
func (o Options) WithUserAgentSuffix(suffix string) Options {
	o = o.clone()
	if suffix == "" {
		return o
	}
	base := o.UserAgent
	if base == "" {
		base = sdkversion.UserAgent()
	}
	o.UserAgent = fmt.Sprintf("%s %s", base, suffix)

	return o
}

// WithMaxResponseBodyBytes returns a new Options instance with the response size cap updated.
func (o Options) WithMaxResponseBodyBytes(n int64) Options {
	o = o.clone()
	o.MaxResponseBodyBytes = n

	return o
}

// WithContext returns a new Options instance with the context updated.
func (o Options) WithContext(ctx context.Context) Options {
	o = o.clone()
	o.ctx = NormalizeContext(ctx)

	return o
}

// Context returns the configured context or context.Background if none was provided.
func (o Options) Context() context.Context {
	return o.ctx
}

// WithDefaultHTTPClient returns a new Options instance that uses the default HTTP client.
func (o Options) WithDefaultHTTPClient() Options {
	o = o.clone()
	o.HTTPClient = DefaultHTTPClient()

	return o
}

// WithHTTPClientFactory returns a new Options instance with an http.Client created by the factory.
// The factory should return a fresh client value; avoid sharing mutable internals like Transport unless synchronized.
// If the factory is nil, the HTTP client is set to nil.
func (o Options) WithHTTPClientFactory(factory func() http.Client) Options {
	o = o.clone()
	if factory == nil {
		o.HTTPClient = nil

		return o
	}
	client := factory()
	o.HTTPClient = &client

	return o
}

// WithHeaders returns a new Options instance with the provided headers applied,
// overriding any existing values for matching keys.
func (o Options) WithHeaders(h http.Header) Options {
	o.Headers = overrideHeaders(o.Headers, h)

	return o
}

// WithHeader returns a new Options instance with a single header applied.
func (o Options) WithHeader(key, value string) Options {
	o.Headers = overrideHeaders(o.Headers, http.Header{key: []string{value}})

	return o
}

// WithDefaultHeader returns a new Options instance with a header set
// only if the header is missing or empty.
func (o Options) WithDefaultHeader(key, value string) Options {
	o.Headers = ensureHeader(o.Headers, key, value)

	return o
}

// clone returns a shallow copy with reference types safely duplicated where possible.
// Headers are deep-copied; context and HTTP client are shared by design.
func (o Options) clone() Options {
	o.Headers = cloneHeader(o.Headers)

	return o
}

// apply applies a series of Option functions to the Options instance.
func (o *Options) apply(opts ...Option) {
	for _, opt := range opts {
		opt(o)
	}
}

// WithBaseURL returns an Option that sets the base URL for API requests.
// The base URL must not include query parameters or fragments.
func WithBaseURL(u string) Option {
	return func(o *Options) {
		o.BaseURL = u
	}
}

// WithToken returns an Option that sets the authentication token for API requests.
func WithToken(t string) Option {
	return func(o *Options) {
		o.Token = t
	}
}

// WithModel returns an Option that sets the model to use for API requests.
func WithModel(m string) Option {
	return func(o *Options) {
		o.Model = m
	}
}

// WithProvider returns an Option that sets the provider for API requests.
func WithProvider(p providers.Provider) Option {
	return func(o *Options) {
		o.Provider = p
	}
}

// WithDefaultProvider returns an Option that sets the provider to the default
// HuggingFace provider.
func WithDefaultProvider() Option {
	return func(o *Options) {
		o.Provider = providers.HuggingFaceProvider{}
	}
}

// WithUserAgent returns an Option that sets the User-Agent header value.
func WithUserAgent(ua string) Option {
	return func(o *Options) {
		o.UserAgent = ua
	}
}

// WithUserAgentSuffix returns an Option that appends a suffix to the SDK user agent string.
func WithUserAgentSuffix(suffix string) Option {
	return func(opts *Options) {
		if suffix == "" {
			return
		}
		base := opts.UserAgent
		if base == "" {
			base = sdkversion.UserAgent()
		}
		opts.UserAgent = fmt.Sprintf("%s %s", base, suffix)
	}
}

// WithMaxResponseBodyBytes returns an Option that sets the maximum response size to read.
func WithMaxResponseBodyBytes(n int64) Option {
	return func(o *Options) {
		o.MaxResponseBodyBytes = n
	}
}

// WithContext returns an Option that sets the context for API requests.
func WithContext(ctx context.Context) Option {
	return func(opts *Options) {
		opts.ctx = NormalizeContext(ctx)
	}
}

// WithDefaultHTTPClient returns an Option that sets the default HTTP client.
func WithDefaultHTTPClient() Option {
	return func(opts *Options) {
		opts.HTTPClient = DefaultHTTPClient()
	}
}

// WithHTTPClientFactory returns an Option that sets a client created by the factory.
// The factory should return a fresh client value; avoid sharing mutable internals like Transport unless synchronized.
// If the factory is nil, the HTTP client is set to nil.
func WithHTTPClientFactory(factory func() http.Client) Option {
	return func(opts *Options) {
		if factory == nil {
			opts.HTTPClient = nil

			return
		}
		client := factory()
		opts.HTTPClient = &client
	}
}

// WithHeaders returns an Option that sets custom headers applied to every request,
// overriding any existing values for matching keys.
// The provided map is copied to avoid unexpected mutations by callers.
func WithHeaders(h http.Header) Option {
	return func(o *Options) {
		o.Headers = overrideHeaders(o.Headers, h)
	}
}

// WithHeader returns an Option that sets a single header applied to every request.
func WithHeader(key, value string) Option {
	return func(o *Options) {
		o.Headers = overrideHeaders(o.Headers, http.Header{key: []string{value}})
	}
}

// WithDefaultHeader returns an Option that sets a header only if missing or empty.
func WithDefaultHeader(key, value string) Option {
	return func(o *Options) {
		o.Headers = ensureHeader(o.Headers, key, value)
	}
}

// cloneHeader returns a deep copy of the provided headers with canonicalized keys.
func cloneHeader(h http.Header) http.Header {
	if len(h) == 0 {
		return nil
	}
	out := make(http.Header, len(h))
	for k, v := range h {
		key := textproto.CanonicalMIMEHeaderKey(k)
		out[key] = append([]string(nil), v...)
	}

	return out
}

// overrideHeaders copies base headers and replaces values with override entries.
// Header keys are canonicalized to avoid duplicate variants.
func overrideHeaders(base, override http.Header) http.Header {
	if len(base) == 0 && len(override) == 0 {
		return nil
	}
	out := cloneHeader(base)
	if out == nil {
		out = make(http.Header, len(override))
	}
	for k, v := range override {
		key := textproto.CanonicalMIMEHeaderKey(k)
		out[key] = append([]string(nil), v...)
	}

	return out
}
