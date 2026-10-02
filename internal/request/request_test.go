//go:build !integration

package request

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfgoversion"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestDo(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		setupOpts   func() hfopts.Options
		method      string
		path        string
		body        io.Reader
		wantErr     bool
		validateReq func(t *testing.T, req *http.Request)
		validateErr func(t *testing.T, err error)
	}{
		{
			name: "builds request correctly",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com"),
					hfopts.WithToken("abc123"),
					hfopts.WithHeaders(http.Header{"X-Test": []string{"yes"}}),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: false,
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				testutils.AssertURL(t, req.URL.String(), &url.URL{
					Scheme: "https",
					Host:   "example.com",
					Path:   "/test",
				})
				if got := req.Header.Get("Authorization"); got != "Bearer abc123" {
					t.Errorf("unexpected Authorization header: %q", got)
				}
				if got := req.Header.Get("X-Test"); got != "yes" {
					t.Errorf("unexpected X-Test header: %q", got)
				}
				if got := req.Header.Get("User-Agent"); got != hfgoversion.UserAgent() {
					t.Errorf(
						"unexpected User-Agent header: %q, want %q",
						got,
						hfgoversion.UserAgent(),
					)
				}
			},
		},
		{
			name: "joins base URL path with relative path",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com/api"),
				)
			},
			method:  http.MethodGet,
			path:    "v1/chat/completions",
			body:    nil,
			wantErr: false,
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				testutils.AssertURL(t, req.URL.String(), &url.URL{
					Scheme: "https",
					Host:   "example.com",
					Path:   "/api/v1/chat/completions",
				})
			},
		},
		{
			name: "preserves query string and fragment",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com/api"),
				)
			},
			method:  http.MethodGet,
			path:    "/v1/chat/completions?model=foo#section",
			body:    nil,
			wantErr: false,
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				testutils.AssertURL(t, req.URL.String(), &url.URL{
					Scheme:   "https",
					Host:     "example.com",
					Path:     "/api/v1/chat/completions",
					RawQuery: "model=foo",
					Fragment: "section",
				})
			},
		},
		{
			name: "context canceled",
			setupOpts: func() hfopts.Options {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithContext(ctx),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				if err == nil {
					t.Fatal("expected context cancellation error")
				}
			},
		},
		{
			name: "nil context uses background",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithContext(testutils.NilContext()),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: false,
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				if req.Context() == nil {
					t.Fatal("expected non-nil request context")
				}
				if err := req.Context().Err(); err != nil {
					t.Fatalf("unexpected context error: %v", err)
				}
			},
		},
		{
			name: "returns API error on non-2xx response",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusUnauthorized, `unauthorized`, nil)

				return hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				}))
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				apiErr := testutils.AssertAPIErrorStatus(t, err, http.StatusUnauthorized)
				if apiErr.Message == "" {
					t.Errorf("expected non-empty API error message")
				}
			},
		},
		{
			name: "header override",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithToken("default"),
					hfopts.WithHeaders(http.Header{"Authorization": []string{"Bearer override"}}),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: false,
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				if got := req.Header.Get("Authorization"); got != "Bearer override" {
					t.Errorf("expected override auth header, got %q", got)
				}
			},
		},
		{
			name: "returns configuration SDKError on bad base URL",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("http://[::1"),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			},
		},
		{
			name: "returns configuration SDKError on base URL with query",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com/api?token=abc"),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			},
		},
		{
			name: "returns configuration SDKError on base URL with fragment",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com/api#section"),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			},
		},
		{
			name: "returns configuration SDKError on base URL with query and fragment",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(
					hfopts.WithHTTPClientFactory(func() http.Client {
						return testutils.NewMockHTTPClient(mt)
					}),
					hfopts.WithBaseURL("https://example.com/api?token=abc#section"),
				)
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			},
		},
		{
			name: "returns internal SDKError on invalid method",
			setupOpts: func() hfopts.Options {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				}))
			},
			method:  "GET\n",
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindInternal)
			},
		},
		{
			name: "returns APIError on nil error response body",
			setupOpts: func() hfopts.Options {
				mt := &testutils.MockTransport{
					Response: &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       nil,
						Header:     make(http.Header),
					},
				}

				return hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				}))
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertAPIErrorStatus(t, err, http.StatusBadRequest)
			},
		},
		{
			name: "returns APIError on http.NoBody error response",
			setupOpts: func() hfopts.Options {
				mt := &testutils.MockTransport{
					Response: &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       http.NoBody,
						Header:     make(http.Header),
					},
				}

				return hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				}))
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertAPIErrorStatus(t, err, http.StatusBadRequest)
			},
		},
		{
			name: "returns configuration SDKError when http client is nil",
			setupOpts: func() hfopts.Options {
				return hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(nil))
			},
			method:  http.MethodGet,
			path:    "/test",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := tt.setupOpts()

			resp, err := Do(opts, tt.method, tt.path, tt.body)

			// Check error expectation
			if (err != nil) != tt.wantErr {
				t.Fatalf("Do() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Validate error if custom validation provided
			if tt.validateErr != nil {
				tt.validateErr(t, err)
			}

			// Validate request if no error and validation provided
			if !tt.wantErr && tt.validateReq != nil {
				// Get the mock transport to access the last request
				if opts.HTTPClient == nil {
					t.Fatal("expected http client")
				}
				if mt, ok := opts.HTTPClient.Transport.(*testutils.MockTransport); ok &&
					mt.LastRequest != nil {
					tt.validateReq(t, mt.LastRequest)
				} else {
					t.Fatal("expected mock transport with LastRequest")
				}
			}

			// Close response body if present
			if resp != nil {
				_ = resp.Body.Close()
			}
		})
	}
}

func TestDoBytes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		data        []byte
		validateReq func(t *testing.T, req *http.Request)
	}{
		{
			name: "sends body correctly",
			data: []byte("hello world"),
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				body, err := io.ReadAll(req.Body)
				if err != nil {
					t.Fatalf("failed to read request body: %v", err)
				}
				if string(body) != "hello world" {
					t.Errorf("unexpected body: %q, want %q", string(body), "hello world")
				}
			},
		},
		{
			name: "sends empty body",
			data: []byte(""),
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				body, err := io.ReadAll(req.Body)
				if err != nil {
					t.Fatalf("failed to read request body: %v", err)
				}
				if len(body) != 0 {
					t.Errorf("unexpected body: %q, want empty", string(body))
				}
			},
		},
		{
			name: "sends JSON data",
			data: []byte(`{"key":"value"}`),
			validateReq: func(t *testing.T, req *http.Request) {
				t.Helper()
				body, err := io.ReadAll(req.Body)
				if err != nil {
					t.Fatalf("failed to read request body: %v", err)
				}
				if !strings.Contains(string(body), "key") {
					t.Errorf("unexpected body: %q", string(body))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)
			opts := hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
				return testutils.NewMockHTTPClient(mt)
			}))

			_, err := DoBytes(opts, http.MethodPost, "/test", tt.data)
			require.NoError(t, err)

			if tt.validateReq != nil && mt.LastRequest != nil {
				tt.validateReq(t, mt.LastRequest)
			}
		})
	}
}

func TestDoRaw(t *testing.T) {
	t.Parallel()

	t.Run("returns response on non-2xx without closing body", func(t *testing.T) {
		tracker := &testutils.CloseTracker{}
		mt := &testutils.MockTransport{
			Response: &http.Response{
				StatusCode: http.StatusUnauthorized,
				Body:       tracker,
				Header:     make(http.Header),
			},
		}
		opts := hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}))

		resp, err := DoRaw(opts, http.MethodGet, "/test", nil)
		require.NoError(t, err)
		if resp == nil || resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("expected status 401 response, got %#v", resp)
		}
		if tracker.Closed {
			t.Fatal("expected response body to remain open")
		}
	})
	t.Run("normalizes nil body to non-nil response body", func(t *testing.T) {
		mt := &testutils.MockTransport{
			Response: &http.Response{
				StatusCode: http.StatusOK,
				Body:       nil,
				Header:     make(http.Header),
			},
		}
		opts := hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
			return testutils.NewMockHTTPClient(mt)
		}))

		resp, err := DoRaw(opts, http.MethodGet, "/test", nil)
		require.NoError(t, err)
		if resp.Body == nil {
			t.Fatal("expected non-nil response body")
		}
	})
	t.Run(
		"returns error when client transport returns nil response without error",
		func(t *testing.T) {
			mt := &testutils.MockTransport{}
			opts := hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
				return testutils.NewMockHTTPClient(mt)
			}))

			_, err := DoRaw(opts, http.MethodGet, "/test", nil)
			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindTransport)
		},
	)
}

func TestJoinURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		baseURL string
		path    string
		want    *url.URL
		wantErr bool
	}{
		{
			name:    "empty path returns base URL",
			baseURL: "https://example.com/api",
			path:    "",
			want: &url.URL{
				Scheme: "https",
				Host:   "example.com",
				Path:   "/api",
			},
		},
		{
			name:    "path only joins with base",
			baseURL: "https://example.com/api",
			path:    "v1/chat",
			want: &url.URL{
				Scheme: "https",
				Host:   "example.com",
				Path:   "/api/v1/chat",
			},
		},
		{
			name:    "query only preserves base path",
			baseURL: "https://example.com/api",
			path:    "?model=foo",
			want: &url.URL{
				Scheme:   "https",
				Host:     "example.com",
				Path:     "/api",
				RawQuery: "model=foo",
			},
		},
		{
			name:    "fragment only preserves base path",
			baseURL: "https://example.com/api",
			path:    "#section",
			want: &url.URL{
				Scheme:   "https",
				Host:     "example.com",
				Path:     "/api",
				Fragment: "section",
			},
		},
		{
			name:    "path with query and fragment",
			baseURL: "https://example.com/api",
			path:    "/v1/chat?model=foo#section",
			want: &url.URL{
				Scheme:   "https",
				Host:     "example.com",
				Path:     "/api/v1/chat",
				RawQuery: "model=foo",
				Fragment: "section",
			},
		},
		{
			name:    "rejects full URL path",
			baseURL: "https://example.com/api",
			path:    "https://evil.example.com/override",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := joinURL(tt.baseURL, tt.path)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected joinURL error")
				}

				return
			}
			if err != nil {
				t.Fatalf("joinURL error: %v", err)
			}
			testutils.AssertURL(t, got, tt.want)
		})
	}
}

func TestDo_IgnoresResponseOnTransportError(t *testing.T) {
	t.Parallel()

	tracker := &testutils.ReadTracker{Data: []byte("ignored")}
	mt := &testutils.MockTransport{
		Response: &http.Response{
			StatusCode: http.StatusOK,
			Body:       tracker,
			Header:     make(http.Header),
		},
		Err: errors.New("boom"),
	}
	opts := hfopts.NewOptions().With(hfopts.WithHTTPClientFactory(func() http.Client {
		return testutils.NewMockHTTPClient(mt)
	}))

	_, err := Do(opts, http.MethodGet, "/test", nil)
	require.Error(t, err)
	if tracker.ReadBytes != 0 {
		t.Fatalf("expected response body to be ignored, read %d bytes", tracker.ReadBytes)
	}
	if tracker.Closed {
		t.Fatal("expected response body to remain open")
	}
}
