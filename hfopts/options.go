package hfopts

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/utils"
	"github.com/Kardbord/hfgo/v4/providers"
	"github.com/Kardbord/hfgo/v4/sdkversion"
)

const (
	// DefaultBaseURL is the default Hugging Face Inference API base URL.
	DefaultBaseURL = "https://router.huggingface.co"
	// DefaultToken is the default (empty) API token.
	DefaultToken = ""
	// DefaultModel is the default (empty) model identifier.
	DefaultModel = ""
	// DefaultMaxResponseBodyBytes caps the amount of response data read into memory by default.
	DefaultMaxResponseBodyBytes int64 = 1 << 20 // 1 MiB
)

// Options captures the configuration shared across all requests in the SDK.
//
// Options are immutable by convention. Applying Option values via With
// returns a new value and defensively clones headers so callers may retain
// or mutate inputs independently.
type Options struct {
	BaseURL string
	Token   string
	Model   string
	// Provider is the inference provider used for endpoint resolution and
	// wire-format transforms. Defaults to HuggingFaceProvider.
	Provider providers.Provider
	// UserAgent is the User-Agent header value sent with requests.
	UserAgent string
	// Headers are custom headers added to every request.
	Headers http.Header
	// HTTPClient is the HTTP client used to send requests.
	HTTPClient *http.Client
	// MaxResponseBodyBytes caps the amount of response data read into memory.
	// Values <= 0 fall back to DefaultMaxResponseBodyBytes.
	MaxResponseBodyBytes int64

	//nolint:containedctx // The context is stored here so that it can be used in request options.
	ctx context.Context
}

// NewOptions returns an Options value initialized with sensible defaults.
func NewOptions() Options {
	defaultClient := DefaultHTTPClient()

	return Options{
		BaseURL:              DefaultBaseURL,
		Token:                DefaultToken,
		Model:                DefaultModel,
		Provider:             providers.NewHuggingFaceProvider(),
		UserAgent:            sdkversion.UserAgent(),
		Headers:              nil,
		HTTPClient:           &defaultClient,
		MaxResponseBodyBytes: DefaultMaxResponseBodyBytes,
		ctx:                  context.Background(),
	}
}

// DefaultHTTPClient returns a fresh default HTTP client value.
func DefaultHTTPClient() http.Client {
	//nolint:exhaustruct // A zero-value http.Client uses the stdlib default transport and no timeout.
	return http.Client{}
}

// Context returns the request-scoped context, defaulting to context.Background().
func (o Options) Context() context.Context {
	if o.ctx != nil {
		return o.ctx
	}

	return context.Background()
}

// With returns a new Options value with the provided options applied.
// Each option is applied in order. Headers are deep-copied so later mutation
// of one value does not affect the other.
func (o Options) With(opts ...Option) Options {
	out := o.clone()
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(&out)
	}

	return out
}

// Validate performs sanity checks on Options and returns an error describing the first issue.
func (o Options) Validate() error {
	if o.HTTPClient == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "http client must not be nil",
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

	parsedURL, err := url.Parse(o.BaseURL)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", o.BaseURL),
			Err:     err,
		}
	}
	if parsedURL.Scheme == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL %q is missing a scheme (e.g. https://)", o.BaseURL),
			Err:     nil,
		}
	}
	if parsedURL.Host == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL %q is missing a host", o.BaseURL),
			Err:     nil,
		}
	}
	if parsedURL.RawQuery != "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "base URL must not contain query parameters",
			Err:     errors.New("query params not allowed in BaseURL"),
		}
	}
	if parsedURL.Fragment != "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "base URL must not contain a fragment",
			Err:     errors.New("fragment not allowed in BaseURL"),
		}
	}

	return nil
}

// clone returns a deep copy of the options header map to prevent mutation.
func (o Options) clone() Options {
	o.Headers = utils.CloneHeader(o.Headers)

	return o
}

// Option mutates request options. Options are applied in order; later options may override earlier ones.
type Option func(*Options)

// WithBaseURL sets the base URL for API requests.
// The base URL must not contain query parameters or a fragment; per-request paths may include query strings.
func WithBaseURL(baseURL string) Option {
	return func(cfg *Options) {
		cfg.BaseURL = baseURL
	}
}

// WithToken sets the HuggingFace API token.
func WithToken(token string) Option {
	return func(cfg *Options) {
		cfg.Token = token
	}
}

// WithModel sets the model for API requests.
func WithModel(model string) Option {
	return func(cfg *Options) {
		cfg.Model = model
	}
}

// WithProvider sets the inference provider used for endpoint resolution and
// wire-format transforms. Passing nil results in a configuration error when
// the options are validated or used.
func WithProvider(provider providers.Provider) Option {
	return func(cfg *Options) {
		cfg.Provider = provider
	}
}

// WithDefaultProvider sets the default HuggingFace provider.
func WithDefaultProvider() Option {
	return func(cfg *Options) {
		cfg.Provider = providers.NewHuggingFaceProvider()
	}
}

// WithUserAgent sets a custom User-Agent header for requests.
func WithUserAgent(userAgent string) Option {
	return func(cfg *Options) {
		cfg.UserAgent = userAgent
	}
}

// WithUserAgentSuffix appends a product token to the SDK user agent when set.
// An empty suffix is a no-op. When the UserAgent is empty, it falls back to
// the default SDK user agent before appending.
func WithUserAgentSuffix(suffix string) Option {
	return func(cfg *Options) {
		suffix = strings.TrimSpace(suffix)
		if suffix == "" {
			return
		}
		base := cfg.UserAgent
		if base == "" {
			base = sdkversion.UserAgent()
		}
		cfg.UserAgent = strings.TrimSpace(base + " " + suffix)
	}
}

// WithMaxResponseBodyBytes sets a cap on the amount of response data read into memory.
// Values <= 0 fall back to DefaultMaxResponseBodyBytes.
func WithMaxResponseBodyBytes(maxBytes int64) Option {
	return func(cfg *Options) {
		if maxBytes <= 0 {
			cfg.MaxResponseBodyBytes = DefaultMaxResponseBodyBytes

			return
		}
		cfg.MaxResponseBodyBytes = maxBytes
	}
}

// WithContext sets the request-scoped context. Nil falls back to context.Background().
func WithContext(ctx context.Context) Option {
	return func(cfg *Options) {
		cfg.ctx = utils.NormalizeContext(ctx)
	}
}

// WithHeaders adds custom headers applied to all requests, replacing any
// existing values for matching keys.
// Headers are deep-copied.
func WithHeaders(h http.Header) Option {
	return func(cfg *Options) {
		cfg.Headers = utils.OverrideHeaders(cfg.Headers, h)
	}
}

// WithHeader sets a single header applied to all requests, replacing any
// existing value for the key.
func WithHeader(key, value string) Option {
	return func(cfg *Options) {
		cfg.Headers = utils.OverrideHeaders(cfg.Headers, http.Header{key: []string{value}})
	}
}

// WithDefaultHeader sets a default header value applied when no header with
// the same key is present.
func WithDefaultHeader(key, value string) Option {
	return func(cfg *Options) {
		cfg.Headers = utils.EnsureHeader(cfg.Headers, key, value)
	}
}

// WithHTTPClientFactory sets a factory for the HTTP client used to send
// requests. The factory is invoked immediately and the returned client is
// stored by pointer. A nil factory results in a nil HTTP client and surfaces
// as a configuration error when the options are used.
//
// HTTP clients are shared across requests, so the factory should return a
// fresh value. Avoid sharing mutable internals like Transport unless they
// are synchronized.
func WithHTTPClientFactory(factory func() http.Client) Option {
	return func(cfg *Options) {
		if factory == nil {
			cfg.HTTPClient = nil

			return
		}
		ensureHTTPClient(cfg, factory)
	}
}

// WithDefaultHTTPClient restores the default HTTP client.
func WithDefaultHTTPClient() Option {
	return func(cfg *Options) {
		ensureDefaultClient(cfg)
	}
}

// ensureHTTPClient assigns the provided HTTP client factory to options.
func ensureHTTPClient(
	options *Options,
	factory func() http.Client,
) {
	if factory == nil {
		options.HTTPClient = nil

		return
	}
	client := factory()
	options.HTTPClient = &client
}

// ensureDefaultClient assigns a fresh default HTTP client value to options.
func ensureDefaultClient(
	options *Options,
) {
	defaultClient := DefaultHTTPClient()
	options.HTTPClient = &defaultClient
}
