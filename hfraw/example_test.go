package hfraw_test

import (
	"context"
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
// The response body is hard-coded so the example can run without a real token;
// copying this into the Go Playground will still hit the mock, not Hugging Face.
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
// endpoint not modeled by the typed SDK. The server returns a hard-coded mock
// response so the example runs without a real token.
func Example() {
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
// preserving the raw HTTP response, even on non-2xx status codes. The 404
// response is hard-coded so the example runs without a real token.
func ExampleClient_DoRaw() {
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

	var apiErr *hferrors.APIError
	if resp.StatusCode == http.StatusNotFound {
		body, _ := io.ReadAll(resp.Body)
		apiErr = &hferrors.APIError{
			StatusCode: resp.StatusCode,
			Message:    http.StatusText(resp.StatusCode),
			Body:       io.NopCloser(strings.NewReader(string(body))),
		}
	}

	fmt.Println(apiErr.StatusCode)
	// Output: 404
}

// ExampleClient_Do_customHeader demonstrates overriding a request header on
// an unimplemented endpoint. The server returns a hard-coded mock response so
// the example runs without a real token.
func ExampleClient_Do_customHeader() {
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
// unimplemented endpoint. The events are hard-coded so the example runs
// without a real token.
func ExampleClient_Stream() {
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
		if err != nil {
			break
		}
		if string(event.Data) == "[DONE]" {
			break
		}
		result.Write(event.Data)
	}

	fmt.Println(result.String())
	// Output: {"message":"Hello"}{"message":" world!"}
}
