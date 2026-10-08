//go:build !integration

package hfraw_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfraw"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestClient_Stream_Success(t *testing.T) {
	t.Parallel()

	body := "data: {\"id\":\"1\"}\n\n" +
		"data: [DONE]\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	stream, err := client.Stream(nil, http.MethodGet, "/stream")
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.JSONEq(t, `{"id":"1"}`, string(event.Data))

	done, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "[DONE]", string(done.Data))
}

func TestClient_Stream_DoError(t *testing.T) {
	t.Parallel()

	mt := &testutils.MockTransport{Err: errors.New("boom")}
	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	_, err := client.Stream(nil, http.MethodGet, "/stream")
	require.Error(t, err)
}

func TestClient_StreamRaw_AllowsNon2xx(t *testing.T) {
	t.Parallel()

	body := "data: hi\n\n"
	mt := testutils.NewMockTransport(http.StatusUnauthorized, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	stream, err := client.StreamRaw(nil, http.MethodGet, "/stream")
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hi", string(event.Data))
}

func TestClient_Do(t *testing.T) {
	t.Parallel()

	t.Run("success returns response", func(t *testing.T) {
		t.Parallel()

		mt := testutils.NewMockTransport(http.StatusOK, `{"ok":true}`, nil)
		client := hfraw.NewClient(
			hfopts.WithHTTPClientFactory(
				func() http.Client { return testutils.NewMockHTTPClient(mt) },
			),
		)

		resp, err := client.Do([]byte(`{"in":1}`), http.MethodPost, "/raw")
		require.NoError(t, err)
		defer func() { _ = resp.Body.Close() }()

		require.Equal(t, http.StatusOK, resp.StatusCode)
		require.NotNil(t, mt.LastRequest)
		require.Equal(t, "/raw", mt.LastRequest.URL.Path)
	})

	t.Run("non-2xx becomes API error", func(t *testing.T) {
		t.Parallel()

		mt := testutils.NewMockTransport(http.StatusUnauthorized, `nope`, nil)
		client := hfraw.NewClient(
			hfopts.WithHTTPClientFactory(
				func() http.Client { return testutils.NewMockHTTPClient(mt) },
			),
		)

		resp, err := client.Do(nil, http.MethodGet, "/raw")
		require.Error(t, err)
		require.Nil(t, resp)

		var apiErr *hferrors.APIError
		require.ErrorAs(t, err, &apiErr)
		require.Equal(t, http.StatusUnauthorized, apiErr.StatusCode)
	})
}

func TestClient_DoRaw(t *testing.T) {
	t.Parallel()

	mt := testutils.NewMockTransport(http.StatusUnauthorized, `nope`, nil)
	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	resp, err := client.DoRaw(nil, http.MethodGet, "/raw")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	require.Equal(t, http.StatusUnauthorized, resp.StatusCode)
}

func TestClient_StreamReader(t *testing.T) {
	t.Parallel()

	body := "data: hi\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	stream, err := client.StreamReader(strings.NewReader("payload"), http.MethodPost, "/stream")
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hi", string(event.Data))
}

func TestClient_StreamRawReader(t *testing.T) {
	t.Parallel()

	body := "data: hi\n\n"
	mt := testutils.NewMockTransport(http.StatusUnauthorized, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	client := hfraw.NewClient(
		hfopts.WithHTTPClientFactory(
			func() http.Client { return testutils.NewMockHTTPClient(mt) },
		),
	)

	stream, err := client.StreamRawReader(strings.NewReader("payload"), http.MethodPost, "/stream")
	require.NoError(t, err)
	defer func() { _ = stream.Close() }()

	event, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hi", string(event.Data))
}

func TestStream_NilReceiver(t *testing.T) {
	t.Parallel()

	var stream hfraw.Stream

	_, err := stream.Recv(context.Background())
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindInternal)

	require.NoError(t, stream.Close())
}
