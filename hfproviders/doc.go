// Package hfproviders maps inference tasks to their HTTP endpoints
// and wire formats.
//
// Two abstractions do the work. A [Provider] resolves an endpoint for
// a task and supplies that task's [Codec]. A [Codec] translates
// between the canonical request and response types in [hftypes] and
// a provider's wire format. The default [HuggingFaceProvider] pairs
// both with the Hugging Face Inference API's endpoints and JSON wire
// format.
//
// # Providers
//
// A [Provider] carries its identity ([Provider.Name] and
// [Provider.ProviderSuffix]) and implements one per-task interface
// for each task it serves, such as [ChatProvider] or
// [TextClassificationProvider]. Each interface pairs an
// endpoint-resolution method (e.g. [ChatProvider.ChatEndpoint]) with
// a codec accessor (e.g. [ChatProvider.ChatCodec]).
//
// Endpoint paths are relative to the configured base URL. The
// transport rejects absolute paths, so a provider cannot redirect a
// request — or its bearer token — to a host the caller did not
// choose. Authentication is caller-owned via options; provider and
// codec implementations must not set an Authorization header.
//
// The typed dispatch layer casts a [Provider] to a per-task interface
// with [AsProvider], returning a configuration error when the cast
// fails.
//
// # Codecs
//
// A [Codec] encodes a canonical request into a wire-format body plus
// headers, and decodes a wire-format response into the canonical
// response type. [JSONCodec] is the JSON implementation.
// [HuggingFaceProvider] composes the built-in per-task mixins, each of
// which supplies that task's endpoint and codec. While streaming, a
// decoder may return [hferrors.EndOfStreamError] to end the stream
// early or [hferrors.SkipEventError] to suppress a frame.
//
// # Extending
//
// Most users do not need to configure a provider. To supply a custom
// one, implement [Provider] plus the per-task interfaces it serves and
// pass it to [hfopts.WithProvider]; [hfopts.WithDefaultProvider]
// restores the built-in default.
//
// To reuse the built-in Hugging Face behavior for a subset of tasks,
// embed the relevant mixins (for example
// [HuggingFaceTextClassificationMixin]) and implement [Provider.Name]
// and [Provider.ProviderSuffix]. Mixins carry no provider identity, so
// composing several is unambiguous.
//
// Partner-provider mixins such as [WavespeedTextToImageMixin] follow the
// same pattern but additionally carry a model resolver in an exported
// field; [WavespeedProvider] composes them and [NewWavespeedProvider]
// wires the resolver into each.
//
// # Model resolution
//
// Third-party providers often serve models under names that differ from
// the Hugging Face Hub. A [ModelResolver] maps a Hugging Face model ID to
// a provider-specific identifier. Providers that need mapping receive a
// resolver as an explicit constructor dependency and consult it in their
// endpoint methods, so lookups happen only inside a provider the caller
// constructed for that purpose. [StaticModelResolver] resolves from a
// caller-supplied mapping with no network I/O, and [HubModelResolver]
// resolves by querying the Hub's inferenceProviderMapping data with
// per-instance caching. Resolution matters only for non-chat tasks; the
// chat endpoint performs provider mapping server-side.
//
// [hfopts.WithProvider]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfopts#WithProvider
// [hfopts.WithDefaultProvider]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfopts#WithDefaultProvider
package hfproviders
