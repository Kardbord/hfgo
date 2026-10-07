//go:build !integration

package request

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// jsonDecode is a decode hook that JSON-unmarshals the event payload
func jsonDecode[T any](_ context.Context, ev RawEvent) (T, error) {
	var out T
	err := json.Unmarshal(ev.Data, &out)

	return out, err
}

func TestJSONStream_InvalidChunk(t *testing.T) {
	t.Parallel()

	body := "data: {not json}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)
	stream := NewJSONStream[map[string]any](raw, jsonDecode[map[string]any])
	defer func() { _ = stream.Close() }()

	_, err = stream.Recv(context.Background())
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
}

func TestJSONStream_NilDecodeGuarded(t *testing.T) {
	t.Parallel()

	body := "data: {\"text\":\"hello\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	// A nil decode hook violates the documented contract; it must surface
	// as an internal error rather than panicking on the call.
	stream := NewJSONStream[map[string]any](raw, nil)
	defer func() { _ = stream.Close() }()

	_, err = stream.Recv(context.Background())
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindInternal)
}

func TestJSONStream_RecvNilContext(t *testing.T) {
	t.Parallel()

	body := "data: {\"text\":\"hello\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)
	type textChunk struct {
		Text string `json:"text"`
	}

	stream := NewJSONStream(raw, jsonDecode[textChunk])
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(testutils.NilContext())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.Text)
}

func TestJSONStream_DecodeHookReceivesNormalizedContext(t *testing.T) {
	t.Parallel()

	body := "data: {\"text\":\"hello\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	type event struct {
		Text string `json:"text"`
	}

	stream := NewJSONStream[event](raw, func(ctx context.Context, ev RawEvent) (event, error) {
		// Panics if ctx is nil; Recv must normalize before the hook runs.
		select {
		case <-ctx.Done():
			return event{}, ctx.Err()
		default:
		}

		var out event
		err := json.Unmarshal(ev.Data, &out)

		return out, err
	})
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(testutils.NilContext())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.Text)
}

func TestJSONStream_DecodeHookReceivesEventName(t *testing.T) {
	t.Parallel()

	body := "event: message_start\ndata: {\"text\":\"hello\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	type event struct {
		Text string `json:"text"`
	}

	var gotEvent string

	stream := NewJSONStream[event](raw, func(_ context.Context, ev RawEvent) (event, error) {
		gotEvent = ev.Event

		var out event
		err := json.Unmarshal(ev.Data, &out)

		return out, err
	})
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.Text)
	require.Equal(t, "message_start", gotEvent)
}

func TestJSONStream_DecodeHookSkipEventError(t *testing.T) {
	t.Parallel()

	body := "event: hidden\ndata: {\"text\":\"phantom\"}\n\n"
	body += "event: token\ndata: {\"text\":\"shown\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	type event struct {
		Text string `json:"text"`
	}

	stream := NewJSONStream[event](raw, func(_ context.Context, ev RawEvent) (event, error) {
		if ev.Event == "hidden" {
			return event{}, hferrors.SkipEventError{}
		}

		var out event
		err := json.Unmarshal(ev.Data, &out)

		return out, err
	})
	defer func() { _ = stream.Close() }()

	// The skipped frame is suppressed; the next one is delivered cleanly.
	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "shown", chunk.Text)
}

func TestJSONStream_DecodeHookEndOfStreamError(t *testing.T) {
	t.Parallel()

	body := "data: {\"text\":\"one\"}\n\ndata: {\"text\":\"never\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	type event struct {
		Text string `json:"text"`
	}

	hookCalls := 0

	stream := NewJSONStream[event](raw, func(_ context.Context, ev RawEvent) (event, error) {
		hookCalls++

		// A wrapped signal must end the stream even though more frames
		// remain in the body.
		return event{}, fmt.Errorf("stop after %q: %w", ev.Event, hferrors.EndOfStreamError{})
	})
	defer func() { _ = stream.Close() }()

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 1, hookCalls)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 1, hookCalls)
}

func TestJSONStream_DoneTrailingFrameDiscarded(t *testing.T) {
	t.Parallel()

	// A frame after the [DONE] sentinel is protocol noise: it must never
	// reach the decode hook or the consumer.
	body := "data: {\"text\":\"one\"}\n\ndata: [DONE]\n\ndata: {\"text\":\"trailing\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	type event struct {
		Text string `json:"text"`
	}

	hookCalls := 0

	stream := NewJSONStream[event](raw, func(_ context.Context, ev RawEvent) (event, error) {
		hookCalls++

		var out event
		err := json.Unmarshal(ev.Data, &out)

		return out, err
	})
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "one", chunk.Text)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)

	_, err = stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 1, hookCalls, "frames after [DONE] must not reach the decoder")
}

func TestJSONStream_SSECloseCancelsRead(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "event: ping\n\n", nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := hfopts.NewOptions().With(
		hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}),
	)

	resp, err := DoBytes(opts, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(opts.Context(), resp.Body)
	require.NoError(t, err)

	stream := NewJSONStream[struct{}](raw, jsonDecode[struct{}])

	go func() {
		time.Sleep(50 * time.Millisecond)
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()

	_, err = stream.Recv(context.Background())
	require.Error(t, err)
}
