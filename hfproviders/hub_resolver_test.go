//go:build !integration

package hfproviders_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

// hubServer simulates the Hugging Face Hub model-info endpoint for resolver
// tests. It counts requests so tests can assert caching behavior, records the
// Authorization header, and serves a configurable status code.
type hubServer struct {
	server   *httptest.Server
	requests atomic.Int32
	payload  atomic.Value // string
	status   atomic.Int32
	auth     atomic.Value // string
}

func newHubServer(t *testing.T, payload string) *hubServer {
	t.Helper()

	hs := &hubServer{}
	hs.payload.Store(payload)
	hs.status.Store(http.StatusOK)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		hs.requests.Add(1)
		hs.auth.Store(r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(int(hs.status.Load()))
		p, _ := hs.payload.Load().(string)
		_, _ = w.Write([]byte(p))
	})

	hs.server = httptest.NewServer(mux)
	t.Cleanup(hs.server.Close)

	return hs
}

func (hs *hubServer) requestCount() int {
	return int(hs.requests.Load())
}

func (hs *hubServer) authHeader() string {
	auth, _ := hs.auth.Load().(string)

	return auth
}

// hubModelInfoPayload is the Hub model-info response used by the option
// wiring tests.
const hubModelInfoPayload = `{
	"id": "black-forest-labs/FLUX.1-dev",
	"inferenceProviderMapping": {
		"fal-ai": {
			"status": "live",
			"providerId": "fal-ai/flux/dev",
			"task": "text-to-image",
			"isModelAuthor": false
		}
	}
}`

// hubResolveParams builds resolver params for model.
func hubResolveParams(model string) hfproviders.ResolveModelParams {
	return hfproviders.ResolveModelParams{
		Model:    model,
		Provider: "fal-ai",
		Task:     "text-to-image",
	}
}

func TestHubModelResolver_ImplementsInterface(t *testing.T) {
	t.Parallel()

	var _ hfproviders.ModelResolver = &hfproviders.HubModelResolver{}
}

func TestHubModelResolver_ResolvesMapping(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "black-forest-labs/FLUX.1-dev",
		"inferenceProviderMapping": {
			"fal-ai": {
				"status": "live",
				"providerId": "fal-ai/flux/dev",
				"task": "text-to-image",
				"isModelAuthor": false
			}
		}
	}`
	hs := newHubServer(t, payload)

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubBaseURL(hs.server.URL))

	mapping, err := resolver.ResolveModel(context.Background(), hfproviders.ResolveModelParams{
		Model:    "black-forest-labs/FLUX.1-dev",
		Provider: "fal-ai",
		Task:     "text-to-image",
	})
	require.NoError(t, err)
	require.Equal(t, hfproviders.ModelMapping{
		ProviderModelID: "fal-ai/flux/dev",
		Adapter:         "",
		Status:          "live",
	}, mapping)
}

func TestHubModelResolver_ResolvesAdapterMapping(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "ZhengmingYu/DMAD",
		"inferenceProviderMapping": {
			"wavespeed": {
				"status": "live",
				"providerId": "wavespeed-ai/minimax-h3/text-to-video-lora",
				"task": "text-to-video",
				"adapter": "lora",
				"adapterWeightsPath": "minimax_h3/dmad.safetensors",
				"isModelAuthor": false
			}
		}
	}`
	hs := newHubServer(t, payload)

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubBaseURL(hs.server.URL))

	mapping, err := resolver.ResolveModel(context.Background(), hfproviders.ResolveModelParams{
		Model:    "ZhengmingYu/DMAD",
		Provider: "wavespeed",
		Task:     "text-to-video",
	})
	require.NoError(t, err)
	require.Equal(t, hfproviders.ModelMapping{
		ProviderModelID:    "wavespeed-ai/minimax-h3/text-to-video-lora",
		Adapter:            "lora",
		AdapterWeightsPath: "minimax_h3/dmad.safetensors",
		Status:             "live",
	}, mapping)
}

func TestHubModelResolver_CachesResults(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "black-forest-labs/FLUX.1-dev",
		"inferenceProviderMapping": {
			"fal-ai": {
				"status": "live",
				"providerId": "fal-ai/flux/dev",
				"task": "text-to-image",
				"isModelAuthor": false
			}
		}
	}`
	hs := newHubServer(t, payload)

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubBaseURL(hs.server.URL))

	params := hfproviders.ResolveModelParams{
		Model:    "black-forest-labs/FLUX.1-dev",
		Provider: "fal-ai",
		Task:     "text-to-image",
	}

	_, err := resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)

	_, err = resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)

	require.Equal(t, 1, hs.requestCount(), "repeated lookups must use the cache")
}

func TestHubModelResolver_AbsentMappingErrors(t *testing.T) {
	t.Parallel()

	payload := `{
		"id": "black-forest-labs/FLUX.1-dev",
		"inferenceProviderMapping": {
			"fal-ai": {
				"status": "live",
				"providerId": "fal-ai/flux/dev",
				"task": "text-to-image",
				"isModelAuthor": false
			}
		}
	}`
	hs := newHubServer(t, payload)

	resolver := hfproviders.NewHubModelResolver(hfproviders.WithHubBaseURL(hs.server.URL))

	_, err := resolver.ResolveModel(context.Background(), hfproviders.ResolveModelParams{
		Model:    "black-forest-labs/FLUX.1-dev",
		Provider: "unmatched-provider",
		Task:     "text-to-image",
	})
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

func TestHubModelResolver_WithHubTokenSendsAuthorization(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubBaseURL(hs.server.URL),
		hfproviders.WithHubToken("test-token"),
	)

	_, err := resolver.ResolveModel(
		context.Background(),
		hubResolveParams("black-forest-labs/FLUX.1-dev"),
	)
	require.NoError(t, err)
	require.Equal(t, "Bearer test-token", hs.authHeader(),
		"WithHubToken must send the token as a Bearer Authorization header")
}

func TestHubModelResolver_NilHTTPClientFactoryIsConfigurationError(t *testing.T) {
	t.Parallel()

	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubHTTPClientFactory(nil),
	)

	_, err := resolver.ResolveModel(
		context.Background(),
		hubResolveParams("black-forest-labs/FLUX.1-dev"),
	)
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}

func TestHubModelResolver_WithHubCacheTTLRefetches(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubBaseURL(hs.server.URL),
		hfproviders.WithHubCacheTTL(time.Millisecond),
	)

	params := hubResolveParams("black-forest-labs/FLUX.1-dev")

	_, err := resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)

	time.Sleep(2 * time.Millisecond)

	_, err = resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)
	require.Equal(t, 2, hs.requestCount(), "an expired entry must refetch")
}

func TestHubModelResolver_WithHubMaxEntriesEvicts(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubBaseURL(hs.server.URL),
		hfproviders.WithHubMaxEntries(1),
		hfproviders.WithHubCacheTTL(time.Minute),
	)

	ctx := context.Background()

	// A cap of one: caching the second model evicts the first.
	for _, model := range []string{"model/a", "model/b"} {
		_, err := resolver.ResolveModel(ctx, hubResolveParams(model))
		require.NoError(t, err)
	}
	require.Equal(t, 2, hs.requestCount())

	_, err := resolver.ResolveModel(ctx, hubResolveParams("model/a"))
	require.NoError(t, err)
	require.Equal(t, 3, hs.requestCount(), "the evicted model must refetch")
}

func TestHubModelResolver_WithHubStaleOnErrorServesStale(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubBaseURL(hs.server.URL),
		hfproviders.WithHubCacheTTL(time.Millisecond),
		hfproviders.WithHubStaleOnError(),
	)

	params := hubResolveParams("black-forest-labs/FLUX.1-dev")

	fresh, err := resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)

	// Expire the entry, then break the Hub.
	time.Sleep(2 * time.Millisecond)
	hs.status.Store(int32(http.StatusInternalServerError))

	stale, err := resolver.ResolveModel(context.Background(), params)
	require.NoError(t, err)
	require.Equal(t, fresh, stale,
		"WithHubStaleOnError must serve the stale mapping when a refresh fails")
}

func TestHubModelResolver_NilOptionIsIgnored(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		nil,
		hfproviders.WithHubBaseURL(hs.server.URL),
	)

	mapping, err := resolver.ResolveModel(
		context.Background(),
		hubResolveParams("black-forest-labs/FLUX.1-dev"),
	)
	require.NoError(t, err)
	require.Equal(t, "fal-ai/flux/dev", mapping.ProviderModelID)
}

func TestHubModelResolver_WithHubHTTPClientFactoryUsesClient(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, hubModelInfoPayload)
	resolver := hfproviders.NewHubModelResolver(
		hfproviders.WithHubBaseURL(hs.server.URL),
		hfproviders.WithHubHTTPClientFactory(func() http.Client {
			return *hs.server.Client()
		}),
	)

	mapping, err := resolver.ResolveModel(
		context.Background(),
		hubResolveParams("black-forest-labs/FLUX.1-dev"),
	)
	require.NoError(t, err)
	require.Equal(t, "fal-ai/flux/dev", mapping.ProviderModelID)
	require.Equal(t, 1, hs.requestCount())
}
