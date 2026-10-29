package hfproviders

import (
	"context"
	"fmt"
	"maps"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ModelResolver maps a Hugging Face model ID to its provider-side
// representation. Third-party providers frequently serve models under
// different names than the Hugging Face Hub does, and provider endpoints need
// that provider-specific identifier at endpoint-construction time.
//
// A resolver is an explicit dependency of the provider that uses it:
// providers that need mapping receive one at construction and call it from
// their endpoint methods (and, for model-in-body wire formats or mapping-
// driven payload changes, from their codecs). Lookups therefore happen only
// inside a provider the caller constructed for that purpose. Resolution
// matters only for non-chat tasks; the chat endpoint performs provider
// mapping server-side.
//
// Implementations must be safe for concurrent use.
type ModelResolver interface {
	// ResolveModel returns the provider-side mapping for the given Hugging
	// Face model ID, provider, and upstream task. The context carries
	// cancellation and deadline signals for the lookup.
	//
	// Implementations must not silently fall back to the original model
	// ID on failure: an unresolvable model should surface as an error.
	ResolveModel(ctx context.Context, params ResolveModelParams) (ModelMapping, error)
}

// ResolveModelParams carries the inputs to [ModelResolver.ResolveModel]. The
// per-call context is passed to ResolveModel separately.
type ResolveModelParams struct {
	// Provider is the partner's Hub identifier, e.g. "fal-ai". The calling
	// provider supplies it; it is the slug the Hub's
	// inferenceProviderMapping data keys mappings by, not necessarily the
	// provider's display name.
	Provider string

	// Task is the upstream task identifier, e.g. [TaskTextToImage] or
	// [TaskConversational]. Provider mappings are task-specific on the Hub.
	Task Task

	// Model is the Hugging Face model ID to resolve, e.g.
	// "black-forest-labs/FLUX.1-dev".
	Model string
}

// ModelMapping is the provider-side representation of a Hugging Face model,
// as declared by the Hub's inferenceProviderMapping data. It mirrors the
// mapping metadata the official Hugging Face clients read, so providers can
// adapt their requests to adapter models without further interface changes.
type ModelMapping struct {
	// ProviderModelID is the provider-specific model identifier, e.g.
	// "fal-ai/flux-1/dev". This is what partner endpoints expect in place
	// of the Hugging Face model ID.
	ProviderModelID string

	// Adapter is the adapter type the Hub mapping declares, e.g. "lora".
	// Some HF repositories ship adapter weights (a small fine-tune applied
	// on top of a provider-hosted base model) rather than a standalone
	// model; empty means the model is not an adapter.
	Adapter string

	// AdapterWeightsPath is the path of the adapter weights inside the
	// Hugging Face repository, e.g. "minimax_h3/dmad.safetensors". Empty
	// when absent. Consumers build provider-appropriate weight URLs from
	// it; formats differ per provider.
	AdapterWeightsPath string

	// Status is the Hub mapping status, e.g. "live" or "staging".
	Status string
}

// StaticMapping identifies a pinned model resolution: the provider's Hub
// identifier, the upstream task, and the Hugging Face model ID.
type StaticMapping struct {
	// Provider is the partner's Hub identifier, e.g. "fal-ai".
	Provider string

	// Task is the upstream task identifier, e.g. [TaskTextToImage].
	Task Task

	// Model is the Hugging Face model ID.
	Model string
}

// StaticModelResolver resolves model IDs from a caller-supplied mapping. It
// performs no network I/O, making it suitable for tests and for deployments
// that pin the HF-to-provider mapping themselves. Entries may carry adapter
// metadata (e.g. [ModelMapping.Adapter]) for models whose provider behavior
// depends on it.
type StaticModelResolver struct {
	mappings map[StaticMapping]ModelMapping
}

// Compile-time assertion that *StaticModelResolver satisfies ModelResolver.
var _ ModelResolver = (*StaticModelResolver)(nil)

// NewStaticModelResolver creates a [StaticModelResolver]. Each entry maps a
// [StaticMapping] key to the provider-side mapping for it. Nil or empty
// mappings are valid; the resolver simply fails every lookup. The map is
// cloned, so callers may retain or mutate it independently.
func NewStaticModelResolver(mappings map[StaticMapping]ModelMapping) *StaticModelResolver {
	return &StaticModelResolver{mappings: maps.Clone(mappings)}
}

// ResolveModel implements [ModelResolver].
func (r *StaticModelResolver) ResolveModel(
	_ context.Context,
	params ResolveModelParams,
) (ModelMapping, error) {
	key := StaticMapping(params)

	resolved, ok := r.mappings[key]
	if !ok {
		return ModelMapping{}, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf(
				"no static mapping for provider %q, task %q, model %q",
				params.Provider, params.Task, params.Model,
			),
			Err: nil,
		}
	}

	return resolved, nil
}
