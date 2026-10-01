//go:build !integration

package request

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestJSONStream_InvalidChunk(t *testing.T) {
	t.Parallel()

	body := "data: {not json}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) })

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

	opts := NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) })

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

func TestJSONStream_SSECloseCancelsRead(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusOK, "event: ping\n\n", nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	opts := NewOptions().
		WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) })

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
