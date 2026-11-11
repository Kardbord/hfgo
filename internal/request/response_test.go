//go:build !integration

package request

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// newStreamParams builds Params whose HTTP client uses the provided mock transport.
func newStreamParams(mt *testutils.MockTransport) Params {
	client := testutils.NewMockHTTPClient(mt)

	return Params{
		HTTPClient: &client,
		BaseURL:    "https://example.com",
	}
}

func TestReadResponseBody(t *testing.T) {
	t.Parallel()

	configKind := hferrors.SDKErrorKindConfiguration
	transportKind := hferrors.SDKErrorKindTransport
	internalKind := hferrors.SDKErrorKindInternal

	tests := []struct {
		name            string
		body            io.Reader
		maxBytes        int64
		wantBody        string
		wantErrContains string
		wantErrIs       error
		wantKind        *hferrors.SDKErrorKind
	}{
		{
			name:     "reads within limit",
			body:     strings.NewReader("hello"),
			maxBytes: 32,
			wantBody: "hello",
		},
		{
			name:            "rejects oversized body",
			body:            strings.NewReader("hello world"),
			maxBytes:        5,
			wantErrContains: "exceeds max size",
			wantKind:        &configKind,
		},
		{
			name:     "non-positive limit falls back to default",
			body:     strings.NewReader("hi"),
			maxBytes: 0,
			wantBody: "hi",
		},
		{
			name: "non-positive limit reports the effective default",
			body: strings.NewReader(
				strings.Repeat("a", int(DefaultMaxResponseBodyBytes)+1),
			),
			maxBytes: 0,
			wantErrContains: fmt.Sprintf(
				"limit %d bytes",
				DefaultMaxResponseBodyBytes,
			),
			wantKind: &configKind,
		},
		{
			name:     "max int64 limit does not overflow",
			body:     strings.NewReader("hi"),
			maxBytes: math.MaxInt64,
			wantBody: "hi",
		},
		{
			name:            "propagates reader error as transport error",
			body:            testutils.ErrorReadCloser{},
			maxBytes:        32,
			wantErrContains: "read failed",
			wantKind:        &transportKind,
		},
		{
			name: "preserves context cancellation",
			body: readerFunc(func([]byte) (int, error) {
				return 0, context.Canceled
			}),
			maxBytes:        32,
			wantErrContains: context.Canceled.Error(),
			wantErrIs:       context.Canceled,
		},
		{
			name: "preserves existing sdk error",
			body: readerFunc(func([]byte) (int, error) {
				return 0, &hferrors.SDKError{
					Kind:    hferrors.SDKErrorKindInternal,
					Message: "reader failed",
				}
			}),
			maxBytes:        32,
			wantErrContains: "reader failed",
			wantKind:        &internalKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resp := &http.Response{Body: io.NopCloser(tt.body)}

			got, err := ReadResponseBody(resp, tt.maxBytes)
			if tt.wantErrContains == "" {
				require.NoError(t, err)
				require.Equal(t, tt.wantBody, string(got))

				return
			}

			require.Error(t, err)
			require.ErrorContains(t, err, tt.wantErrContains)
			if tt.wantErrIs != nil {
				require.ErrorIs(t, err, tt.wantErrIs)
			}
			if tt.wantKind != nil {
				testutils.AssertSDKErrorKind(t, err, *tt.wantKind)
			}
		})
	}
}

func TestReadResponseBodyTruncated_NonPositiveLimitFallsBack(t *testing.T) {
	t.Parallel()

	for _, maxBytes := range []int64{0, -1} {
		body, truncated, err := readResponseBodyTruncated(strings.NewReader("hi"), maxBytes)
		require.NoError(t, err)
		require.False(t, truncated)
		require.Equal(t, "hi", string(body))
	}
}

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(p []byte) (int, error) {
	return f(p)
}

func TestDrainAndCloseBody(t *testing.T) {
	t.Parallel()

	require.NotPanics(t, func() { DrainAndCloseBody(nil, 0) })
	require.NotPanics(t, func() { DrainAndCloseBody(http.NoBody, 0) })

	t.Run("drains small body fully", func(t *testing.T) {
		t.Parallel()

		tracker := &testutils.ReadTracker{Data: []byte("hello")}
		DrainAndCloseBody(tracker, 0)

		require.True(t, tracker.Closed)
		require.Equal(t, len("hello"), tracker.ReadBytes)
	})

	t.Run("uses configured limit when smaller than cap", func(t *testing.T) {
		t.Parallel()

		tracker := &testutils.ReadTracker{Data: []byte("hello world")}
		DrainAndCloseBody(tracker, 5)

		require.True(t, tracker.Closed)
		require.Equal(t, 5, tracker.ReadBytes)
	})

	t.Run("caps drain at maxDrainBytes", func(t *testing.T) {
		t.Parallel()

		tracker := &testutils.ReadTracker{Data: make([]byte, maxDrainBytes+1)}
		DrainAndCloseBody(tracker, DefaultMaxResponseBodyBytes)

		require.True(t, tracker.Closed)
		require.Equal(t, int(maxDrainBytes), tracker.ReadBytes)
	})
}

func TestDecodeHTTPResponse(t *testing.T) {
	t.Parallel()

	configKind := hferrors.SDKErrorKindConfiguration
	serializationKind := hferrors.SDKErrorKindSerialization

	tests := []struct {
		name            string
		statusCode      int
		body            io.Reader
		maxBytes        int64
		wantBody        string
		wantNilBody     bool
		wantErrContains string
		wantKind        *hferrors.SDKErrorKind
	}{
		{
			name:        "204 returns nil body",
			statusCode:  http.StatusNoContent,
			maxBytes:    32,
			wantNilBody: true,
		},
		{
			name:        "205 returns nil body",
			statusCode:  http.StatusResetContent,
			maxBytes:    32,
			wantNilBody: true,
		},
		{
			name:            "empty body is a serialization error",
			statusCode:      http.StatusOK,
			body:            strings.NewReader(""),
			maxBytes:        32,
			wantErrContains: "empty response body",
			wantKind:        &serializationKind,
		},
		{
			name:       "returns body on success",
			statusCode: http.StatusOK,
			body:       strings.NewReader(`{"a":1}`),
			maxBytes:   32,
			wantBody:   `{"a":1}`,
		},
		{
			name:            "propagates oversized body error",
			statusCode:      http.StatusOK,
			body:            strings.NewReader("hello world"),
			maxBytes:        5,
			wantErrContains: "exceeds max size",
			wantKind:        &configKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resp := &http.Response{StatusCode: tt.statusCode}
			if tt.body != nil {
				resp.Body = io.NopCloser(tt.body)
			}

			got, err := DecodeHTTPResponse(resp, tt.maxBytes)
			if tt.wantErrContains == "" {
				require.NoError(t, err)
				if tt.wantNilBody {
					require.Nil(t, got)

					return
				}
				require.Equal(t, tt.wantBody, string(got))

				return
			}

			require.Error(t, err)
			require.ErrorContains(t, err, tt.wantErrContains)
			if tt.wantKind != nil {
				testutils.AssertSDKErrorKind(t, err, *tt.wantKind)
			}
		})
	}
}

func TestValidateEventStreamResponseContentType(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		contentType string
		wantErr     bool
	}{
		{name: "valid", contentType: "text/event-stream", wantErr: false},
		{
			name:        "valid with parameters",
			contentType: "text/event-stream; charset=utf-8",
			wantErr:     false,
		},
		{name: "missing", contentType: "", wantErr: true},
		{name: "wrong mediatype", contentType: "application/json", wantErr: true},
		{name: "malformed", contentType: "text/event-stream; boundary", wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			headers := http.Header{}
			if tc.contentType != "" {
				headers.Set("Content-Type", tc.contentType)
			}

			err := ValidateEventStreamResponseContentType(headers)
			if !tc.wantErr {
				require.NoError(t, err)

				return
			}
			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		})
	}
}

func TestJSONStream_NilRaw(t *testing.T) {
	t.Parallel()

	stream := NewJSONStream[map[string]any](nil, jsonDecode[map[string]any])

	_, err := stream.Recv(context.Background())
	require.Error(t, err)
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindInternal)

	require.Error(t, stream.Close())
}

func TestJSONStream_KeepaliveSkipped(t *testing.T) {
	t.Parallel()

	type event struct {
		Text string `json:"text"`
	}

	body := "event: ping\n\n" + "data: {\"text\":\"hello\"}\n\n"
	mt := testutils.NewMockTransport(http.StatusOK, body, nil)
	mt.Response.Header.Set("Content-Type", "text/event-stream")

	params := newStreamParams(mt)

	resp, err := DoBytes(context.Background(), params, http.MethodPost, "/stream", nil)
	require.NoError(t, err)
	require.NoError(t, ValidateEventStreamResponseContentType(resp.Header))

	raw, err := StreamRaw(context.Background(), resp.Body)
	require.NoError(t, err)

	stream := NewJSONStream[event](raw, jsonDecode[event])
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", chunk.Text)
}
