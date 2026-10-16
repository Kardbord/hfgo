package hfraw_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/Kardbord/hfgo/v4/hferrors"
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

// ExampleClient_Stream_apiError shows how Stream surfaces a non-2xx response
// that is not an event stream. Because a JSON response cannot be consumed as
// SSE, Stream returns an *hferrors.APIError whose Body the caller owns, so the
// error payload is not lost. Extract it with errors.As.
func ExampleClient_Stream_apiError() {
	// The 404 JSON response is hard-coded so the example can run without a real token.
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

	// A JSON (non-event-stream) response cannot be consumed as SSE, so the API
	// error is returned instead of a stream.
	_, err := client.Stream(
		[]byte(`{"stream":true}`),
		http.MethodPost,
		"custom/stream",
	)
	if err == nil {
		panic("expected an error")
	}

	var apiErr *hferrors.APIError
	if !errors.As(err, &apiErr) {
		panic(err)
	}
	defer apiErr.Body.Close()

	body, readErr := io.ReadAll(apiErr.Body)
	if readErr != nil {
		panic(readErr)
	}

	fmt.Println(apiErr.StatusCode, string(body))
	// Output: 404 {"error":"model not found"}
}
