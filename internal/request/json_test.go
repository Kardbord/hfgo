//go:build !integration

package request

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

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
	stream := NewJSONStream[struct{}](raw, nil)
	defer func() { _ = stream.Close() }()

	_, err = stream.Recv(context.Background())
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
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
	stream := NewJSONStream[struct {
		Text string `json:"text"`
	}](raw, nil)
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

	stream := NewJSONStream[struct{}](raw, nil)

	go func() {
		time.Sleep(50 * time.Millisecond)
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()

	_, err = stream.Recv(context.Background())
	require.Error(t, err)
}
