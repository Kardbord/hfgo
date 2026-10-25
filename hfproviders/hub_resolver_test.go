package hfproviders_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/stretchr/testify/require"
)

// hubServer simulates the Hugging Face Hub model-info endpoint for resolver
// tests. It counts requests so tests can assert caching behavior.
type hubServer struct {
	server   *httptest.Server
	requests atomic.Int32
	payload  atomic.Value // string
}

func newHubServer(t *testing.T, payload string) *hubServer {
	t.Helper()

	hs := &hubServer{}
	hs.payload.Store(payload)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, _ *http.Request) {
		hs.requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
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

func TestHubModelResolver_ImplementsInterface(t *testing.T) {
	t.Parallel()

	var _ hfproviders.ModelResolver = &hfproviders.HubModelResolver{}
}

func TestHubModelResolver_ResolvesProviderID(t *testing.T) {
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

	id, err := resolver.ResolveModel(hfproviders.ResolveModelParams{
		Context:  context.Background(),
		Model:    "black-forest-labs/FLUX.1-dev",
		Provider: "fal-ai",
		Task:     "text-to-image",
	})
	require.NoError(t, err)
	require.Equal(t, "fal-ai/flux/dev", id)
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

	_, err := resolver.ResolveModel(params)
	require.NoError(t, err)

	_, err = resolver.ResolveModel(params)
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

	_, err := resolver.ResolveModel(hfproviders.ResolveModelParams{
		Context:  context.Background(),
		Model:    "black-forest-labs/FLUX.1-dev",
		Provider: "unmatched-provider",
		Task:     "text-to-image",
	})
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
}
