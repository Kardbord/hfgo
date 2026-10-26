package hfproviders

import (
	"context"
	"fmt"
	"maps"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// ModelResolver maps a Hugging Face model ID to a provider-specific model
// identifier. Third-party providers frequently serve models under different
// names than the Hugging Face Hub does, and provider endpoints need that
// provider-specific identifier at endpoint-construction time.
//
// A resolver is an explicit dependency of the provider that uses it:
// providers that need mapping receive one at construction and call it from
// their endpoint methods (and, for model-in-body wire formats, from their
// codecs). Lookups therefore happen only inside a provider the caller
// constructed for that purpose. Resolution matters only for non-chat tasks;
// the chat endpoint performs provider mapping server-side.
//
// Implementations must be safe for concurrent use.
type ModelResolver interface {
	// ResolveModel returns the provider-specific model identifier for the
	// given Hugging Face model ID, provider, and upstream task. The context
	// carries cancellation and deadline signals for the lookup.
	//
	// Implementations must not silently fall back to the original model
	// ID on failure: an unresolvable model should surface as an error.
	ResolveModel(ctx context.Context, params ResolveModelParams) (string, error)
}

// ResolveModelParams carries the inputs to [ModelResolver.ResolveModel]. The
// per-call context is passed to ResolveModel separately.
type ResolveModelParams struct {
	// Provider is the partner's Hub identifier, e.g. "fal-ai". The calling
	// provider supplies it; it is the slug the Hub's
	// inferenceProviderMapping data keys mappings by, not necessarily the
	// provider's display name.
	Provider string

	// Task is the upstream task identifier, e.g. "text-to-image" or
	// "conversational". Provider mappings are task-specific on the Hub.
	Task string

	// Model is the Hugging Face model ID to resolve, e.g.
	// "black-forest-labs/FLUX.1-dev".
	Model string
}

// StaticMapping identifies a pinned model resolution: the provider's Hub
// identifier, the upstream task, and the Hugging Face model ID.
type StaticMapping struct {
	// Provider is the partner's Hub identifier, e.g. "fal-ai".
	Provider string

	// Task is the upstream task identifier, e.g. "text-to-image".
	Task string

	// Model is the Hugging Face model ID.
	Model string
}

// StaticModelResolver resolves model IDs from a caller-supplied mapping. It
// performs no network I/O, making it suitable for tests and for deployments
// that pin the HF-to-provider mapping themselves.
type StaticModelResolver struct {
	mappings map[StaticMapping]string
}

// NewStaticModelResolver creates a [StaticModelResolver]. Each entry maps a
// [StaticMapping] key to the provider-specific model identifier. Nil or
// empty mappings are valid; the resolver simply fails every lookup. The map
// is cloned, so callers may retain or mutate it independently.
func NewStaticModelResolver(mappings map[StaticMapping]string) *StaticModelResolver {
	return &StaticModelResolver{mappings: maps.Clone(mappings)}
}

// ResolveModel implements [ModelResolver].
func (r *StaticModelResolver) ResolveModel(
	_ context.Context,
	params ResolveModelParams,
) (string, error) {
	key := StaticMapping(params)

	resolved, ok := r.mappings[key]
	if !ok {
		return "", &hferrors.SDKError{
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
