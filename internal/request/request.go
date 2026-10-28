package request

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// Params carries the transport-level configuration for one HTTP request.
// It is the transport's only view of caller configuration: provider, model,
// and other task-level settings live in hfopts.Options and are projected
// into Params by the layers that own them. The per-request context is passed
// explicitly to the request functions and is not carried by Params.
//
// Params is a value type; Headers is defensively copied when the request is
// built, so callers may retain or mutate their header maps independently.
type Params struct {
	// HTTPClient is the client used to send the request. A nil client is a
	// configuration error reported by [Params.Validate].
	HTTPClient *http.Client

	// BaseURL is the base URL that request paths are joined against.
	BaseURL string

	// UserAgent is sent as the User-Agent header when non-empty.
	UserAgent string

	// Token is sent as a Bearer Authorization header when non-empty.
	Token string

	// Headers are additional request headers applied over the built-in
	// ones. Values for the same key set here replace the built-in values.
	Headers http.Header
}

// Validate performs the transport-level sanity checks that must hold before
// any HTTP request is dispatched: the HTTP client must not be nil, and
// BaseURL must parse with a scheme and host, without query parameters or a
// fragment.
func (p Params) Validate() error {
	return validateTransport(p.HTTPClient, p.BaseURL)
}

// ValidateTransportConfig validates transport-level settings without building
// a Params value. It applies the same checks as [Params.Validate].
func ValidateTransportConfig(httpClient *http.Client, baseURL string) error {
	return validateTransport(httpClient, baseURL)
}

func validateTransport(httpClient *http.Client, baseURL string) error {
	if httpClient == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "http client must not be nil",
			Err:     nil,
		}
	}

	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid base URL %q", baseURL),
			Err:     err,
		}
	}
	if parsedURL.Scheme == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL %q is missing a scheme (e.g. https://)", baseURL),
			Err:     nil,
		}
	}
	if parsedURL.Host == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("base URL %q is missing a host", baseURL),
			Err:     nil,
		}
	}
	if parsedURL.RawQuery != "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "base URL must not contain query parameters",
			Err:     errors.New("query params not allowed in BaseURL"),
		}
	}
	if parsedURL.Fragment != "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "base URL must not contain a fragment",
			Err:     errors.New("fragment not allowed in BaseURL"),
		}
	}

	return nil
}

// UnauthenticatedParams carries the transport configuration for a request
// that must be sent without caller credentials. It deliberately has no Token
// or Headers field, so the caller's bearer token and custom headers cannot be
// attached through it: unauthenticated requests are structurally prevented
// from leaking credentials. The only headers such a request carries are the
// User-Agent and the headers supplied explicitly per call.
type UnauthenticatedParams struct {
	// HTTPClient is the client used to send the request. A nil client is a
	// configuration error reported by [UnauthenticatedParams.Validate].
	HTTPClient *http.Client

	// UserAgent is sent as the User-Agent header when non-empty.
	UserAgent string
}

// Validate performs the transport-level sanity checks that must hold before
// an unauthenticated request is dispatched: the HTTP client must not be nil.
func (p UnauthenticatedParams) Validate() error {
	if p.HTTPClient == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "http client must not be nil",
			Err:     nil,
		}
	}

	return nil
}

// Do performs an HTTP request with the provided context and params and
// returns the response. It creates a new HTTP request with the given method,
// path, and body, adds authorization and custom headers, and executes the
// request using the configured HTTP client.
// For HTTP status codes >= 400, it returns an *hferrors.APIError.
// The caller must close resp.Body on success.
func Do(
	ctx context.Context,
	params Params,
	method string,
	path string,
	body io.Reader,
) (*http.Response, error) {
	resp, err := DoRaw(ctx, params, method, path, body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, APIErrorFromResponse(resp, method)
	}

	return resp, nil
}

// APIErrorFromResponse builds an *hferrors.APIError from a 4xx/5xx response.
// Ownership of resp.Body transfers to the caller, who must close it.
func APIErrorFromResponse(resp *http.Response, method string) *hferrors.APIError {
	reqURL := ""
	if resp.Request != nil && resp.Request.URL != nil {
		reqURL = resp.Request.URL.String()
	}

	return &hferrors.APIError{
		StatusCode: resp.StatusCode,
		Message:    http.StatusText(resp.StatusCode),
		Body:       resp.Body,
		Method:     method,
		URL:        reqURL,
		RequestID:  resp.Header.Get("X-Request-ID"),
	}
}

// DoRaw performs an HTTP request with the provided context and params and
// returns the response without translating 4xx/5xx status codes into SDK
// errors. The caller must close resp.Body on success.
func DoRaw(
	ctx context.Context,
	params Params,
	method string,
	path string,
	body io.Reader,
) (*http.Response, error) {
	// Transport-level checks only: model and provider configuration are
	// task-level concerns enforced by the typed dispatch layer, and raw
	// requests legitimately run without them.
	if err := params.Validate(); err != nil {
		return nil, err
	}

	req, err := buildHTTPRequest(ctx, params, method, path, body)
	if err != nil {
		return nil, err
	}

	return executeRequest(params.HTTPClient, req)
}

func buildHTTPRequest(
	ctx context.Context,
	params Params,
	method, path string,
	body io.Reader,
) (*http.Request, error) {
	ctx = utils.NormalizeContext(ctx)

	reqURL, err := joinURL(params.BaseURL, path)
	if err != nil {
		if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
			return nil, sdkErr
		}

		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("failed to join base URL %q with path %q", params.BaseURL, path),
			Err:     err,
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, reqURL, body)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "failed to create HTTP request",
			Err:     err,
		}
	}

	if params.UserAgent != "" {
		req.Header.Set("User-Agent", params.UserAgent)
	}
	if params.Token != "" {
		req.Header.Set("Authorization", "Bearer "+params.Token)
	}
	req.Header = utils.OverrideHeaders(req.Header, params.Headers)

	return req, nil
}

func executeRequest(client *http.Client, req *http.Request) (*http.Response, error) {
	resp, err := client.Do(req)
	if err != nil {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}

		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindTransport,
			Message: "request failed",
			Err:     err,
		}
	}
	if resp == nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindTransport,
			Message: "http client returned nil response without error",
			Err:     nil,
		}
	}

	if resp.Request == nil {
		resp.Request = req
	}
	if resp.Body == nil {
		resp.Body = http.NoBody
	}

	return resp, nil
}

// joinURL combines a base URL with a relative path while preserving query and fragment.
func joinURL(baseURL, path string) (string, error) {
	if path == "" {
		return baseURL, nil
	}
	parsedPath, err := url.Parse(path)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid path %q", path),
			Err:     err,
		}
	}
	if parsedPath.Scheme != "" || parsedPath.Host != "" {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("path must be relative, got %q", path),
			Err:     nil,
		}
	}
	joined, err := url.JoinPath(baseURL, parsedPath.Path)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf(
				"failed to join base URL %q with path %q",
				baseURL,
				parsedPath.Path,
			),
			Err: err,
		}
	}
	joinedURL, err := url.Parse(joined)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: fmt.Sprintf("failed to parse joined URL %q", joined),
			Err:     err,
		}
	}
	joinedURL.RawQuery = parsedPath.RawQuery
	joinedURL.Fragment = parsedPath.Fragment

	return joinedURL.String(), nil
}

// DoBytes performs an HTTP request with a byte slice body.
// It is a convenience wrapper around Do that converts the byte slice to an io.Reader.
// The caller must close resp.Body on success.
func DoBytes(
	ctx context.Context,
	params Params,
	method string,
	path string,
	data []byte,
) (*http.Response, error) {
	return Do(ctx, params, method, path, bytes.NewReader(data))
}

// DoUnauthenticated performs an HTTP request to an absolute URL without
// applying the caller's credentials. It exists for provider-owned follow-up
// URLs (for example signed CDN output links) that must be fetched without the
// API bearer token. The request carries only the User-Agent and the headers
// supplied explicitly; Authorization is never populated and is stripped
// defensively, and URLs with embedded user credentials are rejected.
// For HTTP status codes >= 400, it returns an *hferrors.APIError.
// The caller must close resp.Body on success.
func DoUnauthenticated(
	ctx context.Context,
	params UnauthenticatedParams,
	method string,
	rawURL string,
	headers http.Header,
	body io.Reader,
) (*http.Response, error) {
	resp, err := DoRawUnauthenticated(ctx, params, method, rawURL, headers, body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, APIErrorFromResponse(resp, method)
	}

	return resp, nil
}

// DoRawUnauthenticated performs an HTTP request to an absolute URL without
// applying the caller's credentials, and returns the response without
// translating 4xx/5xx status codes into SDK errors. See [DoUnauthenticated]
// for the credential rules. The caller must close resp.Body on success.
func DoRawUnauthenticated(
	ctx context.Context,
	params UnauthenticatedParams,
	method string,
	rawURL string,
	headers http.Header,
	body io.Reader,
) (*http.Response, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}

	req, err := buildUnauthenticatedRequest(ctx, params, method, rawURL, headers, body)
	if err != nil {
		return nil, err
	}

	return executeRequest(params.HTTPClient, req)
}

func buildUnauthenticatedRequest(
	ctx context.Context,
	params UnauthenticatedParams,
	method, rawURL string,
	headers http.Header,
	body io.Reader,
) (*http.Request, error) {
	ctx = utils.NormalizeContext(ctx)

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("invalid URL %q", rawURL),
			Err:     err,
		}
	}
	if parsed.Scheme == "" {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("URL must be absolute with a scheme and host, got %q", rawURL),
			Err:     nil,
		}
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("URL scheme must be http or https, got %q", rawURL),
			Err:     nil,
		}
	}
	if parsed.Host == "" {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("URL must include a host, got %q", rawURL),
			Err:     nil,
		}
	}
	// A URL with embedded user credentials (e.g. https://user:pass@host/)
	// would make the transport send a Basic Authorization header. The
	// message deliberately omits the URL so the error cannot echo
	// credentials back to logs.
	if parsed.User != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "URL must not carry user credentials",
			Err:     nil,
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, rawURL, body)
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "failed to create HTTP request",
			Err:     err,
		}
	}

	if params.UserAgent != "" {
		req.Header.Set("User-Agent", params.UserAgent)
	}
	req.Header = utils.OverrideHeaders(req.Header, headers)
	// Defense in depth: the only credential this SDK owns is the caller's
	// bearer token, which must never ride on an unauthenticated request.
	req.Header.Del("Authorization")

	return req, nil
}

// DefaultMaxResponseBodyBytes is the response-body size cap applied when no
// explicit cap is provided. It mirrors hfopts.DefaultMaxResponseBodyBytes,
// which cannot be imported here without an import cycle.
const DefaultMaxResponseBodyBytes int64 = 1 << 20 // 1 MiB

// ReadResponseBody reads the response body up to maxBytes and returns an error
// if the body is larger. If maxBytes is <= 0, DefaultMaxResponseBodyBytes is used.
// The caller is still responsible for closing resp.Body.
func ReadResponseBody(resp *http.Response, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBodyBytes
	}

	bodyBytes, truncated, err := readResponseBodyTruncated(resp.Body, maxBytes)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: fmt.Sprintf("response body exceeds max size (limit %d bytes)", maxBytes),
			Err:     nil,
		}
	}

	return bodyBytes, nil
}

// maxDrainBytes bounds how much of an abandoned response body is drained
// before closing it. Fully draining a body allows the HTTP connection to be
// reused; draining only a bounded amount avoids reading an arbitrarily large
// response after it has already been rejected.
const maxDrainBytes int64 = 32 << 10

// drainLimit returns the number of bytes to drain for the configured maximum
// response body size, capped at maxDrainBytes.
func drainLimit(maxResponseBodyBytes int64) int64 {
	if maxResponseBodyBytes <= 0 {
		maxResponseBodyBytes = DefaultMaxResponseBodyBytes
	}
	if maxResponseBodyBytes > maxDrainBytes {
		return maxDrainBytes
	}

	return maxResponseBodyBytes
}

// DrainAndCloseBody drains at most the configured limit and closes the body.
// The limit is the smaller of maxResponseBodyBytes and maxDrainBytes. If the
// remaining body fits within that budget, the underlying HTTP connection can be
// reused; otherwise it is abandoned.
func DrainAndCloseBody(body io.ReadCloser, maxResponseBodyBytes int64) {
	if body == nil || body == http.NoBody {
		return
	}
	_, _ = io.CopyN(io.Discard, body, drainLimit(maxResponseBodyBytes))
	_ = body.Close()
}

func readResponseBodyTruncated(
	reader io.Reader,
	maxBytes int64,
) (body []byte, truncated bool, err error) {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBodyBytes
	}
	// LimitReader doesn't error on overflow; it just stops at the limit and returns EOF.
	// Read one extra byte so we can detect truncation by checking len(b) > maxBytes.
	limit := maxBytes
	if limit != math.MaxInt64 {
		limit++
	}
	limitReader := io.LimitReader(reader, limit)
	body, err = io.ReadAll(limitReader)
	if err != nil {
		return nil, false, wrapBodyReadError(err)
	}
	if int64(len(body)) > maxBytes {
		return body[:maxBytes], true, nil
	}

	return body, false, nil
}

// wrapBodyReadError classifies an error encountered while reading a response
// body. Context cancellation is preserved so callers can still match it with
// errors.Is, and an existing *hferrors.SDKError is returned unchanged.
func wrapBodyReadError(err error) error {
	if err == nil {
		return nil
	}
	if sdkErr, ok := errors.AsType[*hferrors.SDKError](err); ok {
		return sdkErr
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindTransport,
		Message: "failed to read response body",
		Err:     err,
	}
}
