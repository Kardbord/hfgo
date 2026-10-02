// Package hfraw provides the low-level, advanced escape hatch for the Hugging
// Face Inference API.
//
// It is the deliberate exception to the rest of the SDK. The root Client type
// exposes type-safe inference endpoints for everyday use; hfraw exposes raw
// HTTP and SSE primitives for endpoints the SDK does not model type-safely.
//
// hfraw only routes requests; it does not validate request bodies or interpret
// upstream wire-format quirks. Callers are responsible for paths, bodies, query
// strings, and header handling. Because it mirrors the underlying HTTP API, the
// surface is more likely to change as the upstream API evolves.
//
// Concurrency:
//   - A Client value is immutable and safe for concurrent use by default.
//   - Callers must not mutate request bodies after they are dispatched or while
//     they may be in flight.
package hfraw
