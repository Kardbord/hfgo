package hfproviders

import (
	"context"
	"net/http"
	"time"

	"github.com/Kardbord/hfgo/v4/hfgoversion"
	"github.com/Kardbord/hfgo/v4/internal/hub"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// DefaultHubBaseURL is the default Hugging Face Hub base URL used by
// [HubModelResolver]. It differs from the Inference API's router base URL:
// model mapping lookups go against the Hub host.
const DefaultHubBaseURL = "https://huggingface.co"

// HubModelResolver resolves Hugging Face model IDs to provider-specific model
// identifiers by querying the Hub's inferenceProviderMapping data. It
// implements [ModelResolver]. Results are cached per resolver
// instance, so repeated lookups for the same model do not re-fetch.
//
// A HubModelResolver performs network I/O only when ResolveModel is called;
// constructing one has no side effects. It is safe for concurrent use, and
// its cache is shared by every caller that holds the same instance. The
// resolver holds a pointer to its Hub client, so copying the resolver value
// shares that client and its cache. Sharing is a necessity rather than a
// convenience: the Hub client owns a mutex-guarded cache and must not be
// copied after first use. Copying the resolver value is fine; copying the
// underlying Hub client is not.
type HubModelResolver struct {
	client *hub.Client
}

// Compile-time assertion that *HubModelResolver satisfies ModelResolver.
var _ ModelResolver = (*HubModelResolver)(nil)

// NewHubModelResolver creates a [HubModelResolver] with the provided options.
// The default configuration uses the Hub base URL, no token, a default HTTP
// client, and the documented cache defaults.
func NewHubModelResolver(opts ...HubModelResolverOption) *HubModelResolver {
	//nolint:exhaustruct_v5 // A zero-value http.Client uses the stdlib default transport and no timeout.
	defaultClient := http.Client{}

	//nolint:exhaustruct_v5 // Optional fields are intentionally zero; hub.Client applies documented defaults.
	cfg := hubResolverConfig{
		baseURL:    DefaultHubBaseURL,
		httpClient: &defaultClient,
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(&cfg)
	}

	return &HubModelResolver{
		client: hub.NewClient(
			request.Params{
				HTTPClient: cfg.httpClient,
				BaseURL:    cfg.baseURL,
				UserAgent:  hfgoversion.UserAgent(),
				Token:      cfg.token,
				Headers:    nil,
			},
			hub.Config{
				CacheTTL:     cfg.cacheTTL,
				MaxEntries:   cfg.maxEntries,
				StaleOnError: cfg.staleOnError,
			},
		),
	}
}

// ResolveModel implements [ModelResolver]. The context bounds the Hub lookup,
// including a cache miss's network fetch.
func (r *HubModelResolver) ResolveModel(
	ctx context.Context,
	params ResolveModelParams,
) (string, error) {
	return r.client.ProviderModelID(ctx, params.Provider, string(params.Task), params.Model)
}

type hubResolverConfig struct {
	baseURL      string
	token        string
	httpClient   *http.Client
	cacheTTL     time.Duration
	maxEntries   int
	staleOnError bool
}

// HubModelResolverOption configures a [HubModelResolver] created by
// [NewHubModelResolver].
type HubModelResolverOption func(*hubResolverConfig)

// WithHubBaseURL overrides the Hub base URL used for mapping lookups. When
// this option is not used, lookups target [DefaultHubBaseURL]. A provided
// value is used as-is: an empty string is not replaced with the default and
// fails request validation.
func WithHubBaseURL(baseURL string) HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		cfg.baseURL = baseURL
	}
}

// WithHubToken sets a Hugging Face token sent with Hub lookups. Hub mapping
// endpoints work without a token for public models; a token raises rate
// limits and is required for gated models.
func WithHubToken(token string) HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		cfg.token = token
	}
}

// WithHubHTTPClientFactory sets a factory for the HTTP client used for Hub
// lookups. The factory is invoked immediately and the returned client is
// stored by pointer. A nil factory results in a nil HTTP client and surfaces
// as a configuration error when a lookup is attempted.
//
// Like the SDK's request HTTP client, this client is shared across lookups;
// it must be safe for concurrent use.
func WithHubHTTPClientFactory(factory func() http.Client) HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		if factory == nil {
			cfg.httpClient = nil

			return
		}
		client := factory()
		cfg.httpClient = &client
	}
}

// WithHubCacheTTL sets how long a fetched mapping stays fresh. Values <= 0
// fall back to the client's default TTL.
func WithHubCacheTTL(ttl time.Duration) HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		cfg.cacheTTL = ttl
	}
}

// WithHubMaxEntries caps the number of models the resolver caches,
// evicting least-recently-used models beyond the cap. Values <= 0 fall back
// to the client's default.
func WithHubMaxEntries(maxEntries int) HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		cfg.maxEntries = maxEntries
	}
}

// WithHubStaleOnError makes the resolver serve a stale cached mapping when a
// fresh lookup fails transiently (network error, 5xx, timeout). Client (4xx)
// errors, malformed responses, and cancellation or deadline expiry of the
// lookup's context always propagate instead of serving a stale entry. The
// default is to fail with an error unless a fresh entry is available.
func WithHubStaleOnError() HubModelResolverOption {
	return func(cfg *hubResolverConfig) {
		cfg.staleOnError = true
	}
}
