package hfraw_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfraw"
)

// testClient returns a factory that produces an HTTP client with a dedicated
// transport. This keeps examples isolated from environment proxy settings and
// pointed at the local httptest.Server.
func testClient() func() http.Client {
	return func() http.Client {
		//nolint:exhaustruct // Zero-value Transport uses no proxy and HTTP/1.1.
		return http.Client{Transport: &http.Transport{}}
	}
}

// newMockJSONServer starts a local httptest.Server that always responds with
// 200 OK and Content-Type: application/json, writing body to every request.
// The response body is hard-coded so the example can run without a real token.
// The mock listens on loopback, which the Go Playground permits, so copying
// this helper and an example into the Playground still hits the mock, not
// Hugging Face.
func newMockJSONServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
}

// newMockEventStreamServer starts a local httptest.Server that always responds
// with 200 OK and Content-Type: text/event-stream, writing body to every
// request and flushing it. Like newMockJSONServer, the body is hard-coded.
func newMockEventStreamServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("ResponseWriter does not implement http.Flusher")
		}
		fmt.Fprint(w, body)
		flusher.Flush()
	}))
}

// Example demonstrates using the raw HTTP escape hatch to call an arbitrary
// endpoint not modeled by the typed SDK.
func Example() {
	// The response body is hard-coded so the example can run without a real token.
	server := newMockJSONServer(`{"status":"ok"}`)
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	resp, err := client.Do(
		[]byte(`{"input":"hello"}`),
		http.MethodPost,
		"custom/endpoint",
	)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(body))
	// Output: {"status":"ok"}
}

// ExampleClient_DoRaw demonstrates calling an unimplemented endpoint while
// keeping the raw HTTP response on non-2xx status codes. Unlike Do, DoRaw does
// not translate non-2xx responses into errors, so the status and body are
// yours to inspect.
func ExampleClient_DoRaw() {
	// The 404 response is hard-coded so the example can run without a real token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":"model not found"}`)
	}))
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	resp, err := client.DoRaw(
		[]byte(`{"input":"hello"}`),
		http.MethodPost,
		"custom/endpoint",
	)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%d: %s\n", resp.StatusCode, body)
	// Output: 404: {"error":"model not found"}
}

// ExampleClient_DoReader demonstrates sending a request body from an
// io.Reader. The body is streamed as it is read, so large or generated
// payloads need not be buffered in memory first.
func ExampleClient_DoReader() {
	// The response body is hard-coded so the example can run without a real token.
	server := newMockJSONServer(`{"status":"ok"}`)
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	resp, err := client.DoReader(
		strings.NewReader(`{"input":"hello"}`),
		http.MethodPost,
		"custom/endpoint",
	)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(body))
	// Output: {"status":"ok"}
}

// ExampleClient_DoRawReader is the io.Reader-body counterpart of DoRaw: the
// request body is streamed from a reader, and a non-2xx response is returned
// as data rather than an error.
func ExampleClient_DoRawReader() {
	// The 404 response is hard-coded so the example can run without a real token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":"model not found"}`)
	}))
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	resp, err := client.DoRawReader(
		strings.NewReader(`{"input":"hello"}`),
		http.MethodPost,
		"custom/endpoint",
	)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%d: %s\n", resp.StatusCode, body)
	// Output: 404: {"error":"model not found"}
}

// ExampleClient_Do_customHeader demonstrates overriding a request header on
// an unimplemented endpoint.
func ExampleClient_Do_customHeader() {
	// The response body is hard-coded so the example can run without a real token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom-Header") != "custom-value" {
			w.WriteHeader(http.StatusBadRequest)

			return
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	resp, err := client.Do(
		[]byte(`{"input":"hello"}`),
		http.MethodPost,
		"custom/endpoint",
		hfopts.WithHeader("X-Custom-Header", "custom-value"),
	)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(body))
	// Output: {"status":"ok"}
}

// ExampleClient_Stream demonstrates consuming a raw SSE stream from an
// unimplemented endpoint. Always close the stream when finished.
func ExampleClient_Stream() {
	// The event stream is hard-coded so the example can run without a real token.
	const eventBody = "data: {\"message\":\"Hello\"}\n\n" +
		"data: {\"message\":\" world!\"}\n\n" +
		"data: [DONE]\n\n"

	server := newMockEventStreamServer(eventBody)
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	stream, err := client.Stream(
		[]byte(`{"stream":true}`),
		http.MethodPost,
		"custom/stream",
	)
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	var result strings.Builder
	for {
		event, err := stream.Recv(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		if string(event.Data) == "[DONE]" {
			break
		}
		result.Write(event.Data)
	}

	fmt.Println(result.String())
	// Output: {"message":"Hello"}{"message":" world!"}
}

// ExampleClient_StreamReader consumes an SSE stream while streaming the
// request body from an io.Reader.
func ExampleClient_StreamReader() {
	// The event stream is hard-coded so the example can run without a real token.
	const eventBody = "data: {\"message\":\"Hello\"}\n\n" +
		"data: {\"message\":\" from a reader\"}\n\n" +
		"data: [DONE]\n\n"

	server := newMockEventStreamServer(eventBody)
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	stream, err := client.StreamReader(
		strings.NewReader(`{"stream":true}`),
		http.MethodPost,
		"custom/stream",
	)
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	var result strings.Builder
	for {
		event, err := stream.Recv(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		if string(event.Data) == "[DONE]" {
			break
		}
		result.Write(event.Data)
	}

	fmt.Println(result.String())
	// Output: {"message":"Hello"}{"message":" from a reader"}
}

// ExampleClient_StreamRaw consumes an SSE stream without translating non-2xx
// responses into errors. Where Stream reports a non-2xx status as an error,
// StreamRaw hands the raw events over for you to interpret. This method is for
// advanced users.
func ExampleClient_StreamRaw() {
	// The 404 event stream is hard-coded so the example can run without a real token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "data: {\"error\":\"model not found\"}\n\n")
	}))
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	// The 404 status is not an error here; it arrives in the event payload.
	stream, err := client.StreamRaw(
		[]byte(`{"stream":true}`),
		http.MethodPost,
		"custom/stream",
	)
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	event, err := stream.Recv(context.Background())
	if err != nil {
		panic(err)
	}

	fmt.Println(string(event.Data))
	// Output: {"error":"model not found"}
}

// ExampleClient_StreamRawReader is the io.Reader-body counterpart of
// StreamRaw. Raw events expose the full SSE framing — event name, ID, and
// retry — that typed streams omit.
func ExampleClient_StreamRawReader() {
	// The event stream is hard-coded so the example can run without a real token.
	const eventBody = "event: ping\n" +
		"id: 1\n" +
		"data: {\"message\":\"Hello\"}\n\n"

	server := newMockEventStreamServer(eventBody)
	defer server.Close()

	client := hfraw.NewClient(
		hfopts.WithToken("test-token"),
		hfopts.WithBaseURL(server.URL),
		hfopts.WithHTTPClientFactory(testClient()),
	)

	stream, err := client.StreamRawReader(
		strings.NewReader(`{"stream":true}`),
		http.MethodPost,
		"custom/stream",
	)
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	event, err := stream.Recv(context.Background())
	if err != nil {
		panic(err)
	}

	fmt.Printf("event=%s id=%s data=%s\n", event.Event, event.ID, event.Data)
	// Output: event=ping id=1 data={"message":"Hello"}
}
