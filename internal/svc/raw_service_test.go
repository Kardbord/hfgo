//go:build !integration

package svc_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
)

func TestRawService_Stream_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"1\"}\n\n" +
		"data: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	stream, err := client.Raw().Stream(nil, http.MethodGet, "/stream")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if string(event.Data) != `{"id":"1"}` {
		t.Fatalf("unexpected data: %q", string(event.Data))
	}
	done, err := stream.Recv(context.Background())
	if err != nil {
		t.Fatalf("Recv done: %v", err)
	}
	if string(done.Data) != "[DONE]" {
		t.Fatalf("unexpected done event: %q", string(done.Data))
	}
}

func TestRawService_Stream_DoError(t *testing.T) {
	t.Parallel()

	mt := &testutils.MockTransport{Err: errors.New("boom")}
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	_, err := client.Raw().Stream(nil, http.MethodGet, "/stream")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRawService_StreamRaw_AllowsNon2xx(t *testing.T) {
	t.Parallel()

	body := "data: hi\n\n"
	mt := testutils.NewMockTransport(http.StatusUnauthorized, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	stream, err := client.Raw().StreamRaw(nil, http.MethodGet, "/stream")
	if err != nil {
		t.Fatalf("StreamRaw: %v", err)
	}
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if string(event.Data) != "hi" {
		t.Fatalf("unexpected data: %q", string(event.Data))
	}
}
