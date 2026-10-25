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
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// newTestParams returns Params pointing at the given mock transport with a
// valid base URL; mut tweaks the params before use.
func newTestParams(mt *testutils.MockTransport, mut func(*Params)) Params {
	client := testutils.NewMockHTTPClient(mt)
	params := Params{
		HTTPClient: &client,
		BaseURL:    "https://example.com",
	}
	if mut != nil {
		mut(&params)
	}

	return params
}

func TestDo(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		//nolint:containedctx // The table test carries a per-case request context.
		ctx         context.Context
		setupParams func() Params
		method      string
		path        string
		body        io.Reader
		wantErr     bool
		validateReq func(t *testing.T, req *http.Request)
		validateErr func(t *testing.T, err error)
	}{
		{
			name: "builds request correctly",
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.Token = "abc123"
					p.UserAgent = hfgoversion.UserAgent()
					p.Headers = http.Header{"X-Test": []string{"yes"}}
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "https://example.com/api"
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "https://example.com/api"
				})
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
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()

				return ctx
			}(),
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, nil)
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
			ctx:  testutils.NilContext(),
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, nil)
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
			name: "returns API error on 4xx/5xx response",
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusUnauthorized, `unauthorized`, nil)

				return newTestParams(mt, nil)
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.Token = "default"
					p.Headers = http.Header{"Authorization": []string{"Bearer override"}}
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "http://[::1"
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "https://example.com/api?token=abc"
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "https://example.com/api#section"
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, func(p *Params) {
					p.BaseURL = "https://example.com/api?token=abc#section"
				})
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
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, nil)
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
			setupParams: func() Params {
				mt := &testutils.MockTransport{
					Response: &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       nil,
						Header:     make(http.Header),
					},
				}

				return newTestParams(mt, nil)
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
			setupParams: func() Params {
				mt := &testutils.MockTransport{
					Response: &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       http.NoBody,
						Header:     make(http.Header),
					},
				}

				return newTestParams(mt, nil)
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
			setupParams: func() Params {
				return Params{BaseURL: "https://example.com"}
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
			name: "returns joinURL SDKError unchanged for absolute path",
			setupParams: func() Params {
				mt := testutils.NewMockTransport(http.StatusOK, `{}`, nil)

				return newTestParams(mt, nil)
			},
			method:  http.MethodGet,
			path:    "https://evil.example.com/override",
			body:    nil,
			wantErr: true,
			validateErr: func(t *testing.T, err error) {
				t.Helper()
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
				require.ErrorContains(t, err, "path must be relative")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := tt.setupParams()

			resp, err := Do(tt.ctx, params, tt.method, tt.path, tt.body)

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
				if params.HTTPClient == nil {
					t.Fatal("expected http client")
				}
				if mt, ok := params.HTTPClient.Transport.(*testutils.MockTransport); ok &&
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
			params := newTestParams(mt, nil)

			_, err := DoBytes(context.Background(), params, http.MethodPost, "/test", tt.data)
			require.NoError(t, err)

			if tt.validateReq != nil && mt.LastRequest != nil {
				tt.validateReq(t, mt.LastRequest)
			}
		})
	}
}

func TestDoRaw(t *testing.T) {
	t.Parallel()

	t.Run("returns response on 4xx/5xx without closing body", func(t *testing.T) {
		tracker := &testutils.CloseTracker{}
		mt := &testutils.MockTransport{
			Response: &http.Response{
				StatusCode: http.StatusUnauthorized,
				Body:       tracker,
				Header:     make(http.Header),
			},
		}
		params := newTestParams(mt, nil)

		resp, err := DoRaw(context.Background(), params, http.MethodGet, "/test", nil)
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
		params := newTestParams(mt, nil)

		resp, err := DoRaw(context.Background(), params, http.MethodGet, "/test", nil)
		require.NoError(t, err)
		if resp.Body == nil {
			t.Fatal("expected non-nil response body")
		}
	})
	t.Run(
		"returns error when client transport returns nil response without error",
		func(t *testing.T) {
			mt := &testutils.MockTransport{}
			params := newTestParams(mt, nil)

			_, err := DoRaw(context.Background(), params, http.MethodGet, "/test", nil)
			require.Error(t, err)
			testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindTransport)
		},
	)
}

func TestParamsValidate(t *testing.T) {
	t.Parallel()

	t.Run("accepts valid params", func(t *testing.T) {
		t.Parallel()

		client := testutils.NewMockHTTPClient(testutils.NewMockTransport(http.StatusOK, `{}`, nil))
		params := Params{HTTPClient: &client, BaseURL: "https://example.com"}
		require.NoError(t, params.Validate())
	})
	t.Run("rejects nil http client", func(t *testing.T) {
		t.Parallel()

		params := Params{BaseURL: "https://example.com"}
		err := params.Validate()
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "http client must not be nil")
	})
	t.Run("rejects base URL missing scheme", func(t *testing.T) {
		t.Parallel()

		client := testutils.NewMockHTTPClient(testutils.NewMockTransport(http.StatusOK, `{}`, nil))
		params := Params{HTTPClient: &client, BaseURL: "example.com"}
		err := params.Validate()
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "missing a scheme")
	})
	t.Run("rejects base URL missing host", func(t *testing.T) {
		t.Parallel()

		client := testutils.NewMockHTTPClient(testutils.NewMockTransport(http.StatusOK, `{}`, nil))
		params := Params{HTTPClient: &client, BaseURL: "https://"}
		err := params.Validate()
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "missing a host")
	})
	t.Run("rejects base URL with query", func(t *testing.T) {
		t.Parallel()

		client := testutils.NewMockHTTPClient(testutils.NewMockTransport(http.StatusOK, `{}`, nil))
		params := Params{HTTPClient: &client, BaseURL: "https://example.com?x=1"}
		err := params.Validate()
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "query parameters")
	})
	t.Run("rejects base URL with fragment", func(t *testing.T) {
		t.Parallel()

		client := testutils.NewMockHTTPClient(testutils.NewMockTransport(http.StatusOK, `{}`, nil))
		params := Params{HTTPClient: &client, BaseURL: "https://example.com#f"}
		err := params.Validate()
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "fragment")
	})
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
	params := newTestParams(mt, nil)

	_, err := Do(context.Background(), params, http.MethodGet, "/test", nil)
	require.Error(t, err)
	if tracker.ReadBytes != 0 {
		t.Fatalf("expected response body to be ignored, read %d bytes", tracker.ReadBytes)
	}
	if tracker.Closed {
		t.Fatal("expected response body to remain open")
	}
}

func TestAPIErrorFromResponse(t *testing.T) {
	t.Parallel()

	t.Run("maps response fields", func(t *testing.T) {
		t.Parallel()

		req, err := http.NewRequestWithContext(
			context.Background(),
			http.MethodPost,
			"https://example.com/v1/chat/completions",
			http.NoBody,
		)
		require.NoError(t, err)

		body := io.NopCloser(strings.NewReader(`{"error":"nope"}`))
		resp := &http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       body,
			Header:     http.Header{"X-Request-Id": []string{"req-123"}},
			Request:    req,
		}

		apiErr := APIErrorFromResponse(resp, req.Method)
		require.Equal(t, http.StatusUnauthorized, apiErr.StatusCode)
		require.Equal(t, http.StatusText(http.StatusUnauthorized), apiErr.Message)
		require.Equal(t, body, apiErr.Body)
		require.Equal(t, http.MethodPost, apiErr.Method)
		require.Equal(t, "https://example.com/v1/chat/completions", apiErr.URL)
		require.Equal(t, "req-123", apiErr.RequestID)
	})

	t.Run("tolerates missing request", func(t *testing.T) {
		t.Parallel()

		resp := &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       http.NoBody,
			Header:     http.Header{},
		}

		apiErr := APIErrorFromResponse(resp, http.MethodGet)
		require.Equal(t, http.StatusInternalServerError, apiErr.StatusCode)
		require.Empty(t, apiErr.URL)
	})
}
