// Package hftypes contains the request, response, and parameter types used by
// hfgo's inference methods.
//
// Every inference task has its own types, for example [ChatRequest] and
// [ChatResponse], or [SpeechRecognitionRequest] and [SpeechRecognitionResponse].
// They live here, outside the root package, so the [hfproviders] package can
// reference them without creating an import cycle.
//
// # Concurrency
//
// Client methods receive these types by value and never mutate the caller's
// payload. The copy still shares nested slices, maps, and pointed-to values with
// the caller, so treat a request and the data it references as read-only while a
// call is in flight. To use one request from multiple goroutines, pass a deep
// copy from its Clone method, for example req.Clone().
//
// [hfproviders]: https://pkg.go.dev/github.com/Kardbord/hfgo/v4/hfproviders
package hftypes
