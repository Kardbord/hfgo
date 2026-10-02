// Package hfgo provides Go bindings for the Hugging Face Inference API.
//
// Design notes:
//   - The public surface is defined in internal sub-packages and re-exported
//     here by type alias. Internal packages (internal/dto, internal/task,
//     internal/chatstream, internal/request, internal/hferrors,
//     internal/sdkversion) are implementation details; import them only from
//     within the module. The providers package is public and defines pluggable
//     inference providers: wire-format encode/decode and endpoint resolution.
//   - Clients are immutable; options are fixed at creation time, and each call snapshots them.
//   - Clients are safe for concurrent use by default or when configured with immutable or synchronized dependencies.
//   - Per-request options can override client defaults for a single call.
//   - Request options are applied by value with defensive header copies; contexts and HTTP clients are shared.
//   - HTTP client injection uses a value factory; return a fresh client value to avoid shared state.
//   - The SDK favors upstream feature parity and uses DTOs closely aligned to the API; breaking changes are possible as the upstream API evolves.
//   - WithDefaultHTTPClient restores the default client; a nil factory is treated as a configuration error.
//   - Client.Raw() returns the RawClient escape hatch for arbitrary endpoints, exposing both error-interpreting and raw request paths (Do vs DoRaw).
//   - DTO validation is enforced during JSON marshal/unmarshal. Invalid request
//     payloads surface as configuration errors. For responses, invalid content
//     type surfaces as validation errors, while malformed JSON surfaces as
//     serialization errors.
//   - Concurrency assumes externally supplied objects (for example, transports) are not mutated after use
//     unless callers provide their own synchronization.
//   - Request DTOs are passed to Client methods by value and the SDK never mutates the caller's payload.
//   - A request and the nested data it references must be treated as read-only while a call is in flight.
//     For concurrent invocation, pass a defensive copy per call (for example, go client.Chat(req.Clone(), ...))
//     or build a fresh request per call. Every request DTO provides a deep Clone method.
package hfgo
