//go:build !integration

package hub_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/hub"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/stretchr/testify/require"
)

// hubServer simulates the Hugging Face Hub model-info endpoint. It counts
// requests so tests can assert caching behavior.
type hubServer struct {
	server   *httptest.Server
	requests atomic.Int32
	payload  atomic.Value // string
	status   atomic.Int32
}

func newHubServer(t *testing.T, payload string) *hubServer {
	t.Helper()

	hs := &hubServer{}
	hs.payload.Store(payload)
	hs.status.Store(http.StatusOK)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, _ *http.Request) {
		hs.requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(int(hs.status.Load()))
		payload, _ := hs.payload.Load().(string)
		_, _ = w.Write([]byte(payload))
	})

	hs.server = httptest.NewServer(mux)
	t.Cleanup(hs.server.Close)

	return hs
}

func (hs *hubServer) setPayload(payload string) {
	hs.payload.Store(payload)
}

func (hs *hubServer) requestCount() int {
	return int(hs.requests.Load())
}

// newTestClient constructs a [hub.Client] pointed at the mock server.
func newTestClient(t *testing.T, hs *hubServer, cfg hub.Config) *hub.Client {
	t.Helper()

	params := request.Params{
		BaseURL:    hs.server.URL,
		HTTPClient: hs.server.Client(),
	}

	return hub.NewClient(params, cfg)
}

// mappingObjectShape mirrors the model_info response: inferenceProviderMapping
// is a dict keyed by provider slug, and the provider name comes from the key.
const mappingObjectShape = `{
  "_id": "66aaa908fc35e079a941470d",
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

// mappingKeyedDictShape is the model_info shape with several providers,
// including one with a non-live status.
const mappingKeyedDictShape = `{
  "_id": "66aaa908fc35e079a941470d",
  "id": "black-forest-labs/FLUX.1-dev",
  "inferenceProviderMapping": {
    "fal-ai": {
      "status": "live",
      "providerId": "fal-ai/flux/dev",
      "task": "text-to-image",
      "isModelAuthor": false
    },
    "replicate": {
      "status": "live",
      "providerId": "black-forest-labs/flux-dev",
      "task": "text-to-image",
      "isModelAuthor": false
    },
    "deepinfra": {
      "status": "error",
      "providerId": "black-forest-labs/FLUX-1-dev",
      "task": "text-to-image",
      "isModelAuthor": false
    }
  }
}`

const mappingListShape = `{
  "_id": "6503d98f2f9a9a92f1a5b3c1",
  "id": "black-forest-labs/FLUX.1-dev",
  "inferenceProviderMapping": [
    {
      "provider": "fal-ai",
      "providerId": "fal-ai/flux/dev",
      "task": "text-to-image",
      "status": "live"
    },
    {
      "provider": "replicate",
      "providerId": "black-forest-labs/flux-dev",
      "task": "text-to-image",
      "status": "live"
    },
    {
      "provider": "novita",
      "provider_id": "black-forest-labs/flux-dev",
      "task": "text-to-image",
      "status": "staging"
    }
  ]
}`

// mappingDuplicateProviderShape repeats a provider with different tasks,
// which is why mappings are kept as a slice rather than keyed by provider.
const mappingDuplicateProviderShape = `{
  "id": "some/model",
  "inferenceProviderMapping": [
    {
      "provider": "fal-ai",
      "providerId": "fal-ai/text-to-image",
      "task": "text-to-image",
      "status": "live"
    },
    {
      "provider": "fal-ai",
      "providerId": "fal-ai/conversational",
      "task": "conversational",
      "status": "live"
    }
  ]
}`

func TestMappings_ObjectShape(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "black-forest-labs/FLUX.1-dev")
	require.NoError(t, err)
	require.Len(t, mappings, 1)
	require.Equal(t, "fal-ai", mappings[0].Provider)
	require.Equal(t, "fal-ai/flux/dev", mappings[0].ProviderModelID)
	require.Equal(t, "text-to-image", mappings[0].Task)
	require.Equal(t, "live", mappings[0].Status)
}

func TestMappings_KeyedDictShape(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingKeyedDictShape)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "black-forest-labs/FLUX.1-dev")
	require.NoError(t, err)
	require.Len(t, mappings, 3)

	// The provider name comes from the dict key, and iteration order is
	// deterministic (sorted by provider slug).
	require.Equal(t, "deepinfra", mappings[0].Provider)
	require.Equal(t, "black-forest-labs/FLUX-1-dev", mappings[0].ProviderModelID)
	require.Equal(t, "error", mappings[0].Status)
	require.Equal(t, "fal-ai", mappings[1].Provider)
	require.Equal(t, "fal-ai/flux/dev", mappings[1].ProviderModelID)
	require.Equal(t, "replicate", mappings[2].Provider)
	require.Equal(t, "black-forest-labs/flux-dev", mappings[2].ProviderModelID)
}

func TestMappings_KeyedDictSkipsMalformedEntry(t *testing.T) {
	t.Parallel()

	payload := `{
		"inferenceProviderMapping": {
			"good": {
				"providerId": "good/model",
				"task": "text-to-image",
				"status": "live"
			},
			"bad": "not-an-object"
		}
	}`
	hs := newHubServer(t, payload)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "some/model")
	require.NoError(t, err)
	require.Len(t, mappings, 1)
	require.Equal(t, "good", mappings[0].Provider)
	require.Equal(t, "good/model", mappings[0].ProviderModelID)
}

func TestMappings_ListShape(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingListShape)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "black-forest-labs/FLUX.1-dev")
	require.NoError(t, err)
	require.Len(t, mappings, 3)
	require.Equal(t, "fal-ai", mappings[0].Provider)
	require.Equal(t, "fal-ai/flux/dev", mappings[0].ProviderModelID)
	require.Equal(t, "replicate", mappings[1].Provider)
	require.Equal(t, "black-forest-labs/flux-dev", mappings[1].ProviderModelID)
	// snake_case spellings are tolerated.
	require.Equal(t, "novita", mappings[2].Provider)
	require.Equal(t, "staging", mappings[2].Status)
}

func TestMappings_CachesWithinTTL(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{CacheTTL: time.Minute})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	first, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	second, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	require.Equal(t, 1, hs.requestCount(), "a fresh cached entry must not trigger a fetch")
	require.Equal(t, first, second)
}

func TestMappings_ExpiryRefetches(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{CacheTTL: 250 * time.Millisecond})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	first, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Wait for the entry to expire, then change what the server serves.
	// The generous TTL leaves ample stall budget for the re-cache
	// assertion below on loaded runners.
	time.Sleep(300 * time.Millisecond)
	hs.setPayload(mappingKeyedDictShape)

	second, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	require.Equal(t, 2, hs.requestCount(), "an expired entry must trigger a refetch")
	require.NotEqual(t, first, second, "a refetch must return fresh data, not the expired entry")

	// The fresh data must be re-cached: an immediate lookup is served from
	// the cache.
	third, err := client.Mappings(ctx, model)
	require.NoError(t, err)
	require.Equal(t, 2, hs.requestCount(), "a refetched entry must be re-cached")
	require.Equal(t, second, third)
}

func TestMappings_StaleOnError(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		StaleOnError: true,
	})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	fresh, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then break the server.
	time.Sleep(2 * time.Millisecond)
	hs.setPayload(`{}`)
	hs.status.Store(int32(http.StatusInternalServerError))

	stale, err := client.Mappings(ctx, model)
	require.NoError(t, err)
	require.Equal(t, fresh, stale,
		"StaleOnError must serve the stale entry when a fresh fetch fails")
}

func TestMappings_NoStaleOnErrorPropagatesFailure(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		StaleOnError: false,
	})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	_, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then break the server.
	time.Sleep(2 * time.Millisecond)
	hs.setPayload(`{}`)
	hs.status.Store(int32(http.StatusInternalServerError))

	_, err = client.Mappings(ctx, model)
	require.Error(t, err)

	var apiErr *hferrors.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, http.StatusInternalServerError, apiErr.StatusCode)
}

func TestMappings_StaleOnErrorClientErrorPropagates(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		StaleOnError: true,
	})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	_, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then fail with a deterministic client error: a 404
	// is not transient, so it must propagate rather than serving stale.
	time.Sleep(2 * time.Millisecond)
	hs.status.Store(int32(http.StatusNotFound))

	_, err = client.Mappings(ctx, model)
	require.Error(t, err)

	var apiErr *hferrors.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, http.StatusNotFound, apiErr.StatusCode)
}

func TestMappings_StaleOnErrorMalformedResponsePropagates(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		StaleOnError: true,
	})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	_, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then serve a malformed mapping payload: a decode
	// failure is not transient and must propagate rather than serving
	// stale.
	time.Sleep(2 * time.Millisecond)
	hs.setPayload(`{"inferenceProviderMapping": "not-a-shape"}`)

	_, err = client.Mappings(ctx, model)
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
}

func TestMappings_StaleOnErrorClientTimeout(t *testing.T) {
	t.Parallel()

	// The first request is served instantly; subsequent requests are slow
	// enough to trip the client's timeout.
	var requests atomic.Int32

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, _ *http.Request) {
		if requests.Add(1) > 1 {
			time.Sleep(100 * time.Millisecond)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mappingObjectShape))
	})

	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)

	httpClient := server.Client()

	client := hub.NewClient(
		request.Params{BaseURL: server.URL, HTTPClient: httpClient},
		hub.Config{CacheTTL: time.Millisecond, StaleOnError: true},
	)

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	// No client timeout yet: under a loaded race-detector run even the
	// instant first request could trip a tight budget.
	first, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then give the client a tight timeout that the slow
	// refresh will trip. The caller's context is fine, so the stale entry
	// is served.
	time.Sleep(2 * time.Millisecond)
	httpClient.Timeout = 20 * time.Millisecond

	stale, err := client.Mappings(ctx, model)
	require.NoError(t, err)
	require.Equal(t, first, stale)
}

func TestMappings_ContextCancellation(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.Mappings(ctx, "black-forest-labs/FLUX.1-dev")
	require.Error(t, err)
}

func TestMappings_MissingMappingField(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, `{"id": "some/model"}`)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "some/model")
	require.NoError(t, err)
	require.Empty(t, mappings, "a model with no inferenceProviderMapping data has no mappings")

	// Selection on a model with no mappings is a configuration error.
	_, err = client.ProviderModelID(context.Background(), "fal-ai", "text-to-image", "some/model")
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
	require.Contains(t, sdkErr.Message, "fal-ai")
}

func TestMappings_InvalidJSON(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, `{"inferenceProviderMapping": "not-a-shape"`)
	client := newTestClient(t, hs, hub.Config{})

	_, err := client.Mappings(context.Background(), "some/model")
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
}

func TestProviderModelID_SelectsMatchingMapping(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingListShape)
	client := newTestClient(t, hs, hub.Config{})

	id, err := client.ProviderModelID(
		context.Background(),
		"fal-ai",
		"text-to-image",
		"black-forest-labs/FLUX.1-dev",
	)
	require.NoError(t, err)
	require.Equal(t, "fal-ai/flux/dev", id)
}

func TestProviderModelID_SelectsTaskAmongDuplicateProviders(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingDuplicateProviderShape)
	client := newTestClient(t, hs, hub.Config{})

	id, err := client.ProviderModelID(
		context.Background(),
		"fal-ai",
		"conversational",
		"some/model",
	)
	require.NoError(t, err)
	require.Equal(t, "fal-ai/conversational", id)
}

func TestProviderModelID_ToleratesSnakeCaseProviderID(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingListShape)
	client := newTestClient(t, hs, hub.Config{})

	id, err := client.ProviderModelID(
		context.Background(),
		"novita",
		"text-to-image",
		"black-forest-labs/FLUX.1-dev",
	)
	require.NoError(t, err)
	require.Equal(t, "black-forest-labs/flux-dev", id)
}

func TestProviderModelID_NoMatchingProvider(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingListShape)
	client := newTestClient(t, hs, hub.Config{})

	_, err := client.ProviderModelID(
		context.Background(),
		"nonexistent",
		"text-to-image",
		"black-forest-labs/FLUX.1-dev",
	)
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
	require.Contains(t, sdkErr.Message, "nonexistent")
}

func TestProviderModelID_NoMatchingTask(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingListShape)
	client := newTestClient(t, hs, hub.Config{})

	_, err := client.ProviderModelID(
		context.Background(),
		"fal-ai",
		"conversational",
		"black-forest-labs/FLUX.1-dev",
	)
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindConfiguration, sdkErr.Kind)
	// The error names the task the mapping actually serves.
	require.Contains(t, sdkErr.Message, "text-to-image")
}

func TestMappings_ConcurrentAccess(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		MaxEntries:   4,
		StaleOnError: true,
	})

	models := []string{"a", "b", "c", "d"}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for i := range 50 {
				_, err := client.Mappings(
					context.Background(),
					models[i%len(models)],
				)
				if err != nil {
					t.Errorf("Mappings: %v", err)

					return
				}
			}
		})
	}

	// Change the payload mid-run so refresh paths return different data.
	time.Sleep(5 * time.Millisecond)
	hs.setPayload(mappingKeyedDictShape)

	wg.Wait()
}

func TestMappings_StaleOnErrorContextCancellation(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		CacheTTL:     time.Millisecond,
		StaleOnError: true,
	})

	ctx := context.Background()
	model := "black-forest-labs/FLUX.1-dev"

	_, err := client.Mappings(ctx, model)
	require.NoError(t, err)

	// Expire the entry, then fetch with a canceled context: cancellation
	// must propagate rather than serving the stale entry.
	time.Sleep(2 * time.Millisecond)

	canceled, cancel := context.WithCancel(ctx)
	cancel()

	_, err = client.Mappings(canceled, model)
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
}

func TestMappings_LRUEviction(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		MaxEntries: 2,
		CacheTTL:   time.Minute,
	})

	ctx := context.Background()

	// Fill the cache with three models; the first must be evicted.
	for _, model := range []string{"a", "b", "c"} {
		_, err := client.Mappings(ctx, "models/"+model)
		require.NoError(t, err)
	}

	// "a" was evicted, so fetching it again must trigger a new request.
	_, err := client.Mappings(ctx, "models/a")
	require.NoError(t, err)

	require.Equal(t, 4, hs.requestCount())
}

func TestMappings_LRURefreshMovesToFront(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{
		MaxEntries: 2,
		CacheTTL:   50 * time.Millisecond,
	})

	ctx := context.Background()

	// Fill the cache with "a" then "b"; "a" becomes least recently used.
	_, err := client.Mappings(ctx, "models/a")
	require.NoError(t, err)
	_, err = client.Mappings(ctx, "models/b")
	require.NoError(t, err)

	// Let both entries expire, then refresh "a". A refresh must mark the
	// entry most recently used so "b" is evicted next, not "a".
	time.Sleep(60 * time.Millisecond)
	_, err = client.Mappings(ctx, "models/a")
	require.NoError(t, err)

	// Adding "c" evicts the least recently used entry, "b".
	_, err = client.Mappings(ctx, "models/c")
	require.NoError(t, err)

	// "a" is still cached: this lookup must not trigger a request.
	before := hs.requestCount()
	_, err = client.Mappings(ctx, "models/a")
	require.NoError(t, err)
	require.Equal(t, before, hs.requestCount(), "refreshed entry must remain cached")

	// "b" was evicted: this lookup must trigger a request.
	_, err = client.Mappings(ctx, "models/b")
	require.NoError(t, err)
	require.Equal(t, before+1, hs.requestCount(), "least recently used entry must be evicted")
}

func TestDecodeMappings_TolerantFieldSpellings(t *testing.T) {
	t.Parallel()

	payload := `{"inferenceProviderMapping": [{"provider": "p", "provider_id": "pid", "task": "t", "status": "s"}]}`
	hs := newHubServer(t, payload)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "m")
	require.NoError(t, err)
	require.Len(t, mappings, 1)
	require.Equal(t, "pid", mappings[0].ProviderModelID)
}

func TestMappings_PayloadShapeIsJSON(t *testing.T) {
	t.Parallel()

	hs := newHubServer(t, mappingObjectShape)
	client := newTestClient(t, hs, hub.Config{})

	mappings, err := client.Mappings(context.Background(), "m")
	require.NoError(t, err)
	require.NotEmpty(t, mappings)
}
