// This example demonstrates how to observe the HTTP traffic the SDK sends by
// wrapping the client's http.Transport with a logging [http.RoundTripper].
//
// The SDK deliberately stays quiet at runtime: logging configuration (backend,
// verbosity, redaction policy) is a caller decision, so the SDK exposes
// hfopts.WithHTTPClientFactory instead of a built-in logger. Wrapping the
// transport gives you request-level visibility — method, URL, status, latency
// — for every call the SDK makes, with any logging backend you choose.
//
// To keep this example free to run (no HF_TOKEN, no inference call, no
// network), it points hfopts.WithBaseURL at a local test server instead of
// the real Hugging Face router. The pattern works unchanged with the default
// base URL and any provider.
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// redactedAuthorization is what the logging transport prints in place of a
// real bearer token. The token never reaches the log: it is replaced before
// any attribute is emitted.
const redactedAuthorization = "[REDACTED]"

// loggingTransport is an [http.RoundTripper] that logs each request's method,
// URL, status code, and duration. It never mutates the in-flight request and
// never logs request or response bodies: the Authorization header carries the
// caller's bearer token, and chat bodies carry prompt content, so neither
// belongs in a log line. Wrap more layers (retries, metrics, tracing) the
// same way if you need them.
type loggingTransport struct {
	// inner is the transport to delegate to. A nil inner falls back to
	// [http.DefaultTransport], which is what a zero-value http.Client uses.
	inner http.RoundTripper
	// log receives one structured record per request.
	log *slog.Logger
}

// RoundTrip implements [http.RoundTripper]. It clones the request's header
// map before logging so the original request is untouched while in flight,
// per the SDK's request-safety rules.
func (t loggingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	inner := t.inner
	if inner == nil {
		inner = http.DefaultTransport
	}

	start := time.Now()

	hasAuthorization := req.Header.Get("Authorization") != ""

	resp, err := inner.RoundTrip(req)

	attrs := []slog.Attr{
		slog.String("method", req.Method),
		slog.String("url", req.URL.String()),
		slog.Bool("authorization_present", hasAuthorization),
		slog.Int64("duration_ms", time.Since(start).Milliseconds()),
	}

	switch {
	case err != nil:
		attrs = append(attrs, slog.Any("error", err))
		t.log.Error("http request failed", attrsToAny(attrs)...)

		return nil, err // preserve the original response/error contract
	default:
		attrs = append(attrs, slog.Int("status", resp.StatusCode))
		t.log.Info("http request", attrsToAny(attrs)...)
	}

	return resp, nil
}

// attrsToAny adapts slog attributes to the variadic key/value form that
// Logger.Info and Logger.Error accept.
func attrsToAny(attrs []slog.Attr) []any {
	out := make([]any, 0, len(attrs))
	for _, attr := range attrs {
		out = append(out, attr)
	}

	return out
}

func main() {
	if err := run(); err != nil {
		log.Fatalf("http-logging example failed: %v", err)
	}
}

// run performs the whole example so that main stays defer-free: an
// os.Exit-class failure in main (log.Fatalf) would otherwise skip deferred
// cleanup such as the test server shutdown.
func run() error {
	// A local test server stands in for the Hugging Face router so the
	// example makes no real API call and needs no token. Its response is a
	// minimal OpenAI-compatible chat completion, which is the shape the SDK
	// decodes into hftypes.ChatResponse.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "demo-1",
			"created": 1735689600,
			"model": "openai/gpt-oss-120b",
			"choices": [{
				"finish_reason": "stop",
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Bonjour! The capital of France is Paris."
				}
			}],
			"usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19}
		}`)
	}))
	defer server.Close()

	logger := slog.Default()

	// Wire the logging transport into the client. The token below is a
	// placeholder: the example never contacts the real API, and the transport
	// redacts it from logs regardless.
	client := hfgo.NewClient(
		hfopts.WithBaseURL(server.URL),
		hfopts.WithToken("hf_placeholder_not_a_real_token"),
		hfopts.WithModel("openai/gpt-oss-120b"),
		hfopts.WithHTTPClientFactory(func() http.Client {
			return http.Client{
				Transport: loggingTransport{
					inner: http.DefaultTransport,
					log:   logger,
				},
			}
		}),
	)

	request := hftypes.ChatRequest{
		Messages: []hftypes.ChatMessage{
			{
				Role: "user",
				Content: hftypes.ChatMessageContent{
					Text: new("Hello! What is the capital of France?"),
				},
			},
		},
		MaxTokens: new(1024),
	}

	response, err := client.Chat(context.Background(), request)
	if err != nil {
		return fmt.Errorf("chat request failed: %w", err)
	}

	// The call itself is unaffected by the transport: same request, same
	// response. Only the log output is new.
	fmt.Println("Chat Completion Response:")
	fmt.Printf("  ID: %s\n", response.ID)
	fmt.Printf("  Model: %s\n", response.Model)

	for _, choice := range response.Choices {
		if choice.Message.Content != nil {
			fmt.Printf("  Content: %s\n", *choice.Message.Content)
		}
	}

	return nil
}
