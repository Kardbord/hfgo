// Package hftypes contains the request, response, and parameter types used by
// hfgo's inference methods.
//
// Request types use the upstream task noun plus Request, for example
// [SummarizationRequest]. Response types use the result noun when one exists,
// for example [Summarization] or [QuestionAnswer]. The Response suffix is
// reserved for whole response envelopes whose bare name would be a task/verb or
// collide, for example [ChatResponse], [ChatStreamResponse],
// [SpeechRecognitionResponse], and [TextToImageResponse]. Value and element
// types do not take the Response suffix.
//
// These types live here, outside the root package, so the [hfproviders] package
// can reference them without creating an import cycle.
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
