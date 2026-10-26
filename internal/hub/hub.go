// Package hub resolves model IDs via the Hugging Face Hub's
// inferenceProviderMapping data. It is the transport used by
// [hfproviders.HubModelResolver] and is not part of the public SDK surface.
package hub

import (
	"bytes"
	"container/list"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// DefaultCacheTTL is how long a fetched mapping stays fresh when the client
// is constructed with a zero CacheTTL.
const DefaultCacheTTL = 5 * time.Minute

// DefaultMaxEntries is the number of models cached when the client is
// constructed with a zero MaxEntries.
const DefaultMaxEntries = 128

// Mapping is one inference-provider mapping entry for a model, as reported
// by the Hugging Face Hub's inferenceProviderMapping data.
type Mapping struct {
	// Provider is the partner's Hub identifier, e.g. "fal-ai".
	Provider string

	// ProviderModelID is the provider-specific model identifier, e.g.
	// "fal-ai/flux-1/dev". This is what partner endpoints expect in place
	// of the Hugging Face model ID.
	ProviderModelID string

	// Task is the upstream task this mapping serves, e.g. "text-to-image".
	Task string

	// Status is the mapping status ("live", "staging", ...). The client
	// does not filter on status; selection reports whatever the Hub
	// returns.
	Status string
}

// Config holds optional tuning knobs for a [Client].
type Config struct {
	// CacheTTL is how long a fetched mapping stays fresh. Zero uses
	// [DefaultCacheTTL].
	CacheTTL time.Duration

	// MaxEntries is the maximum number of models cached (LRU eviction).
	// Zero uses [DefaultMaxEntries].
	MaxEntries int

	// StaleOnError serves a stale cache entry when a fresh fetch fails
	// transiently (network error, 5xx, 429 rate limiting, timeout). Other
	// client (4xx) errors, malformed responses, and cancellation or deadline
	// expiry of the lookup's context always propagate instead. When false
	// (the default), all fetch errors propagate even if a stale entry
	// exists.
	StaleOnError bool
}

// cacheEntry is one cached model's mappings plus its LRU bookkeeping. All
// fields are guarded by the owning [Client]'s mutex.
type cacheEntry struct {
	// mappings are the provider mappings returned by the Hub for the model.
	// They are kept as a slice rather than an index keyed by provider: a
	// model has only a handful of entries, the same provider may appear more
	// than once with different tasks, and preserving the Hub's order keeps
	// selection deterministic. ProviderModelID scans this short slice
	// linearly. The slice and its backing array are never mutated after
	// decode; a refresh replaces the field with a new slice.
	mappings []Mapping

	// expiry is the instant after which the entry is stale and must be
	// fetched again.
	expiry time.Time

	// element is this entry's node in the owning Client's LRU list. The
	// list holds the model string as its value.
	element *list.Element
}

// Client resolves Hugging Face model IDs to provider-specific model
// identifiers via the Hub's inferenceProviderMapping data. It is an
// implementation detail of [hfproviders.HubModelResolver] and is not part of
// the public SDK surface.
//
// A Client is safe for concurrent use. It must not be copied after first use:
// copying duplicates the mutex's internal state, so the copy no longer
// synchronizes with the original's cache. Copying is invalid, not merely
// discouraged; the compiler permits it, but go vet's copylocks analyzer flags
// it. [NewClient] returns a pointer so callers share one client rather than
// copying it.
type Client struct {
	// params is the transport configuration used for Hub requests.
	params request.Params

	// ttl is how long a fetched mapping stays fresh. NewClient replaces a
	// non-positive Config.CacheTTL with [DefaultCacheTTL].
	ttl time.Duration

	// maxEntries caps the number of models in the cache. NewClient replaces
	// a non-positive Config.MaxEntries with [DefaultMaxEntries].
	maxEntries int

	// staleOnError serves a stale entry when a refresh fails transiently.
	staleOnError bool

	// mu guards entries and lru, and serializes cache refresh decisions.
	mu sync.Mutex

	// entries maps a Hub model ID to its cache entry.
	entries map[string]*cacheEntry

	// lru orders models from most to least recently used. Its values are
	// model strings, and every entry in entries has exactly one element
	// here.
	lru *list.List
}

// NewClient creates a [Client]. Params carries the Hub transport
// configuration: BaseURL must point at the Hub host (e.g.
// "https://huggingface.co"), which differs from the Inference API's router
// base URL. Token is optional; unauthenticated Hub requests may serve
// reduced data.
//
// A zero Config selects the documented defaults.
func NewClient(params request.Params, cfg Config) *Client {
	ttl := cfg.CacheTTL
	if ttl <= 0 {
		ttl = DefaultCacheTTL
	}
	maxEntries := cfg.MaxEntries
	if maxEntries <= 0 {
		maxEntries = DefaultMaxEntries
	}

	//nolint:exhaustruct_v5 // mu is a zero-value mutex; it needs no initialization.
	return &Client{
		params:       params,
		ttl:          ttl,
		maxEntries:   maxEntries,
		staleOnError: cfg.StaleOnError,
		entries:      make(map[string]*cacheEntry),
		lru:          list.New(),
	}
}

// ProviderModelID returns the provider-specific model identifier for the
// given Hub model ID, provider, and upstream task. A model with no mapping
// for the provider is a configuration error; a provider mapping that serves
// a different task is a configuration error naming the supported task.
func (c *Client) ProviderModelID(
	ctx context.Context,
	provider, task, model string,
) (string, error) {
	mappings, err := c.Mappings(ctx, model)
	if err != nil {
		return "", err
	}

	var supported []string
	for _, mapping := range mappings {
		if mapping.Provider != provider {
			continue
		}
		if mapping.Task == task {
			return mapping.ProviderModelID, nil
		}
		supported = append(supported, mapping.Task)
	}

	if len(supported) > 0 {
		return "", &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf(
				"model %q mapping for provider %q serves task %q, not %q",
				model, provider, strings.Join(supported, `", "`), task,
			),
			Err: nil,
		}
	}

	return "", &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindConfiguration,
		Message: fmt.Sprintf("model %q has no inference mapping for provider %q", model, provider),
		Err:     nil,
	}
}

// Mappings returns all inference-provider mappings for the given Hub model
// ID, using and refreshing the cache as needed.
func (c *Client) Mappings(ctx context.Context, model string) ([]Mapping, error) {
	c.mu.Lock()
	entry, exists := c.entries[model]

	var cached []Mapping
	var expiry time.Time
	if exists {
		c.lru.MoveToFront(entry.element)
		// Snapshot while holding the lock: a concurrent refresh replaces
		// these fields in place. The backing array is never mutated after
		// decode, so the snapshot stays valid after the lock is released.
		cached, expiry = entry.mappings, entry.expiry
	}
	c.mu.Unlock()

	if exists && time.Now().Before(expiry) {
		return cached, nil
	}

	fetched, err := c.fetch(ctx, model)
	if err != nil {
		// Stale entries cover transient upstream failures only;
		// deterministic problems and caller cancellation propagate.
		if c.staleOnError && exists && isTransientFetchFailure(ctx, err) {
			// A 5xx or 429 failure surfaces as an *hferrors.APIError that
			// owns the open response body. Nothing will read that body once
			// the error is discarded, so close it here.
			if apiErr, ok := errors.AsType[*hferrors.APIError](err); ok {
				request.DrainAndCloseBody(apiErr.Body, 0)
			}

			return cached, nil
		}

		return nil, err
	}

	return c.cacheFetch(model, fetched, entry, exists), nil
}

// isTransientFetchFailure reports whether a fetch failure is a transient
// upstream condition worth serving a stale entry for: a transport error, a
// 5xx Hub response, or a 429 rate-limit response. A done caller context —
// cancellation or deadline expiry — always propagates, as do other 4xx
// responses and malformed payloads.
func isTransientFetchFailure(ctx context.Context, err error) bool {
	if ctx.Err() != nil {
		return false
	}

	if apiErr, ok := errors.AsType[*hferrors.APIError](err); ok {
		return apiErr.IsServerError() || apiErr.IsRateLimitError()
	}

	if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
		return sdkErr.Kind == hferrors.SDKErrorKindTransport
	}

	return false
}

// cacheFetch stores fetched mappings for model and returns the mappings the
// caller should use: a concurrent goroutine's fresher entry when one
// appeared during the fetch, otherwise the fetched mappings themselves.
// observed is the (possibly expired) entry seen before the fetch, and exists
// reports whether there was one.
func (c *Client) cacheFetch(
	model string,
	fetched []Mapping,
	observed *cacheEntry,
	exists bool,
) []Mapping {
	c.mu.Lock()
	defer c.mu.Unlock()

	other, present := c.entries[model]
	switch {
	case present && (!exists || other != observed):
		// A concurrent goroutine refreshed first; prefer its entry.
		c.lru.MoveToFront(other.element)

		return other.mappings
	case present:
		// The expired entry we observed is still current: refresh it in
		// place, reusing its LRU element so the list and map stay
		// consistent.
		other.mappings = fetched
		other.expiry = time.Now().Add(c.ttl)
		c.lru.MoveToFront(other.element)
	default:
		// No entry now (there was none, or ours was evicted mid-fetch).
		entry := &cacheEntry{
			mappings: fetched,
			expiry:   time.Now().Add(c.ttl),
			element:  c.lru.PushFront(model),
		}
		c.entries[model] = entry
		c.evictLocked()
	}

	return fetched
}

// evictLocked removes the least-recently-used entries beyond maxEntries.
// Callers must hold c.mu.
func (c *Client) evictLocked() {
	for len(c.entries) > c.maxEntries {
		oldest := c.lru.Back()
		if oldest == nil {
			break
		}

		if model, ok := oldest.Value.(string); ok {
			delete(c.entries, model)
		}
		c.lru.Remove(oldest)
	}
}

// fetch retrieves the inferenceProviderMapping data for the model from the
// Hub and decodes it tolerantly: the Hub serves this field as a dict keyed
// by provider on model_info and as a list of mappings on list_models.
func (c *Client) fetch(ctx context.Context, model string) ([]Mapping, error) {
	path := "/api/models/" + model + "?expand[]=inferenceProviderMapping"

	//nolint:bodyclose // The body is drained and closed after decoding.
	resp, err := request.Do(ctx, c.params, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}

	body, err := request.ReadResponseBody(resp, 0)
	request.DrainAndCloseBody(resp.Body, 0)
	if err != nil {
		return nil, err
	}

	mappings, err := decodeMappings(body)
	if err != nil {
		return nil, err
	}

	return mappings, nil
}

type hubModelInfo struct {
	//nolint:tagliatelle // The Hub serves this field in camelCase.
	InferenceProviderMapping json.RawMessage `json:"inferenceProviderMapping"`
}

// mappingEntry is one raw inferenceProviderMapping object. The Hub has served
// the provider model identifier under both camelCase and snake_case spellings,
// so both are accepted. Unknown fields (for example isModelAuthor) are ignored.
type mappingEntry struct {
	Provider string `json:"provider"`

	//nolint:tagliatelle // The Hub serves this field in camelCase.
	ProviderModelIDCamel string `json:"providerId"`

	ProviderModelIDSnake string `json:"provider_id"`

	Task   string `json:"task"`
	Status string `json:"status"`
}

// mapping converts a raw entry to a Mapping. It returns false when the
// provider or provider model identifier is missing.
func (e mappingEntry) mapping() (Mapping, bool) {
	providerID := e.ProviderModelIDCamel
	if providerID == "" {
		providerID = e.ProviderModelIDSnake
	}
	if e.Provider == "" || providerID == "" {
		return Mapping{}, false
	}

	return Mapping{
		Provider:        e.Provider,
		ProviderModelID: providerID,
		Task:            e.Task,
		Status:          e.Status,
	}, true
}

// inferenceProviderMapping is the union of the two shapes the Hub serves for
// the inferenceProviderMapping field: a dict keyed by provider slug on
// model_info and a list of self-describing entries on list_models (see
// huggingface_hub PR #3022). Its UnmarshalJSON normalizes either shape into a
// flat []Mapping. Field names are matched tolerantly because upstream
// spellings have varied (camelCase and snake_case).
type inferenceProviderMapping []Mapping

// UnmarshalJSON accepts either the model_info dict or the list_models array.
func (m *inferenceProviderMapping) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		*m = nil

		return nil
	}

	var keyed map[string]json.RawMessage
	if err := json.Unmarshal(data, &keyed); err == nil {
		*m = decodeKeyedMappings(keyed)

		return nil
	}

	var listed []json.RawMessage
	if err := json.Unmarshal(data, &listed); err != nil {
		return fmt.Errorf("unexpected inferenceProviderMapping shape: %w", err)
	}
	*m = decodeListedMappings(listed)

	return nil
}

// decodeKeyedMappings converts the model_info dict shape into mappings,
// taking each provider name from its key. Entries that fail to decode are
// skipped.
func decodeKeyedMappings(keyed map[string]json.RawMessage) []Mapping {
	mappings := make([]Mapping, 0, len(keyed))
	for _, provider := range slices.Sorted(maps.Keys(keyed)) {
		var entry mappingEntry
		if err := json.Unmarshal(keyed[provider], &entry); err != nil {
			continue
		}
		// The provider name comes from the dict key, mirroring upstream's
		// {**value, "provider": provider}.
		entry.Provider = provider
		if mapping, ok := entry.mapping(); ok {
			mappings = append(mappings, mapping)
		}
	}

	return mappings
}

// decodeListedMappings converts the list_models array shape into mappings.
// Entries that fail to decode are skipped.
func decodeListedMappings(listed []json.RawMessage) []Mapping {
	mappings := make([]Mapping, 0, len(listed))
	for _, raw := range listed {
		var entry mappingEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			continue
		}
		if mapping, ok := entry.mapping(); ok {
			mappings = append(mappings, mapping)
		}
	}

	return mappings
}

// decodeMappings parses the Hub model-info response. The
// inferenceProviderMapping field is a dict keyed by provider slug on
// model_info — the provider name comes from the key, not the entry — and a
// list of self-describing entries on list_models. Field names are matched
// tolerantly because upstream spellings have varied (camelCase and
// snake_case).
func decodeMappings(body []byte) ([]Mapping, error) {
	var info hubModelInfo
	if err := json.Unmarshal(body, &info); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode Hub model info",
			Err:     err,
		}
	}
	// A model with no inferenceProviderMapping data simply has no provider
	// mappings; selection reports that as a configuration error.
	if len(info.InferenceProviderMapping) == 0 {
		return nil, nil
	}

	var mappings inferenceProviderMapping
	if err := json.Unmarshal(info.InferenceProviderMapping, &mappings); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "unexpected inferenceProviderMapping shape in Hub response",
			Err:     err,
		}
	}

	return mappings, nil
}
