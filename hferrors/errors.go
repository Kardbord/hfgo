package hferrors

import (
	"fmt"
	"net/http"
	"time"

	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// APIError represents an error returned by the Hugging Face API.
// It includes the HTTP status code, error message, response body, response
// headers, method, and URL. For client-side errors produced before
// an API response, see [SDKError].
type APIError struct {
	// StatusCode is the HTTP status code returned by the API
	StatusCode int

	// Message is the human-readable error message. It carries the HTTP status
	// text followed by the (truncated) response body when one was available,
	// e.g. "Not Found: {\"error\":\"Model not found\"}". The body portion is
	// capped for log-friendly output, and control characters (including
	// newlines) are removed so server-controlled bytes cannot forge log lines;
	// the full body is available in [APIError.Body].
	Message string

	// Body contains the full response body from the API as buffered bytes,
	// capped by the request's MaxResponseBodyBytes option (1 MiB by default).
	// It is nil when no response body was available. The body is read eagerly
	// when the error is constructed; callers never need to close anything.
	// Treat the slice as read-only.
	Body []byte

	// Headers contains the response headers as returned by the API, with
	// canonicalized keys, deep-copied so the error never aliases the live
	// response. It is nil when no response headers were available. It never
	// contains request headers, so credentials are never exposed here.
	Headers http.Header

	// Method is the HTTP method used for the request
	Method string

	// URL is the URL that was requested
	URL string
}

// retryAfterCap caps the server-provided Retry-After delay reported by
// [APIError.RetryAfter] so a pathological header cannot stall a caller's
// retry loop for an unbounded time.
const retryAfterCap = 60 * time.Second

// Error implements the error interface for APIError.
func (e *APIError) Error() string {
	return fmt.Sprintf("hf api error (%d): %s", e.StatusCode, e.Message)
}

// RetryAfter returns the server-provided retry delay from the Retry-After
// response header, capped at one minute. It returns 0 when the header is
// absent, does not parse, or holds a non-positive value.
func (e *APIError) RetryAfter() time.Duration {
	return utils.ParseRetryAfter(e.Headers, retryAfterCap)
}

// IsClientError returns true if the error is a 4xx client error.
func (e *APIError) IsClientError() bool {
	return e.StatusCode >= 400 && e.StatusCode < 500
}

// IsServerError returns true if the error is a 5xx server error.
func (e *APIError) IsServerError() bool {
	return e.StatusCode >= 500 && e.StatusCode < 600
}

// IsAuthenticationError returns true if the error is a 401 Unauthorized error.
func (e *APIError) IsAuthenticationError() bool {
	return e.StatusCode == http.StatusUnauthorized
}

// IsRateLimitError returns true if the error is a 429 Too Many Requests error.
func (e *APIError) IsRateLimitError() bool {
	return e.StatusCode == http.StatusTooManyRequests
}

// SDKErrorKind represents the category of a client-side SDK error.
// These errors are produced locally (not returned by the API).
type SDKErrorKind string

const (
	// SDKErrorKindValidation indicates a validation error in API responses.
	SDKErrorKindValidation SDKErrorKind = "validation"
	// SDKErrorKindConfiguration indicates invalid or missing configuration.
	SDKErrorKindConfiguration SDKErrorKind = "configuration"
	// SDKErrorKindSerialization indicates a serialization or deserialization error.
	SDKErrorKindSerialization SDKErrorKind = "serialization"
	// SDKErrorKindTransport indicates a transport-layer failure.
	SDKErrorKindTransport SDKErrorKind = "transport"
	// SDKErrorKindProvider indicates a failure the provider reported inside an
	// otherwise valid API response, such as a remote task reaching a terminal
	// failure status. The HTTP exchange itself succeeded; for HTTP-level
	// failures see [APIError].
	SDKErrorKindProvider SDKErrorKind = "provider"
	// SDKErrorKindInternal indicates an internal SDK error.
	SDKErrorKindInternal SDKErrorKind = "internal"
)

// SDKError represents a client-side SDK error that occurred before
// a response was received from the API, or while trying to unmarshal
// the response from the API. For errors returned by the API itself,
// see [APIError].
type SDKError struct {
	// Kind is the category of error (configuration/serialization/etc).
	Kind SDKErrorKind

	// Message is the human-readable error message.
	Message string

	// Err is the underlying error, if any.
	Err error
}

// Error implements the error interface for SDKError.
func (e *SDKError) Error() string {
	if e.Err != nil && e.Message != "" {
		return fmt.Sprintf("sdk error (%s): %s: %v", e.Kind, e.Message, e.Err)
	}
	if e.Err != nil {
		return fmt.Sprintf("sdk error (%s): %v", e.Kind, e.Err)
	}
	if e.Message != "" {
		return fmt.Sprintf("sdk error (%s): %s", e.Kind, e.Message)
	}

	return fmt.Sprintf("sdk error (%s)", e.Kind)
}

// Unwrap returns the underlying error, if any.
func (e *SDKError) Unwrap() error {
	return e.Err
}
