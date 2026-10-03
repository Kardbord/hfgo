package hferrors

import (
	"fmt"
	"io"
	"net/http"
)

// APIError represents an error returned by the HuggingFace API.
// It includes the HTTP status code, error message, response body,
// and request ID if available.
type APIError struct {
	// StatusCode is the HTTP status code returned by the API
	StatusCode int

	// Message is the human-readable error message
	Message string

	// Body contains the response body from the API as a readable stream.
	// The caller is responsible for closing this body to ensure proper resource cleanup.
	// If Body is nil, no response body was available.
	Body io.ReadCloser

	// RequestID is the request identifier from the X-Request-ID header, if available
	RequestID string

	// Method is the HTTP method used for the request
	Method string

	// URL is the URL that was requested
	URL string
}

// Error implements the error interface for APIError.
func (e *APIError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf(
			"hf api error (%d): %s [request_id: %s]",
			e.StatusCode,
			e.Message,
			e.RequestID,
		)
	}

	return fmt.Sprintf("hf api error (%d): %s", e.StatusCode, e.Message)
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
	// SDKErrorKindInternal indicates an internal SDK error.
	SDKErrorKindInternal SDKErrorKind = "internal"
)

// SDKError represents a client-side SDK error that occurred before
// a response was received from the API, or while trying to unmarshal
// the response from the API.
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
