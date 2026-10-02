//go:build !integration

package hferrors

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAPIError_Error(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  *APIError
	}{
		{
			name: "basic error message",
			err: &APIError{
				StatusCode: http.StatusBadRequest,
				Message:    "Bad Request",
			},
		},
		{
			name: "error with request ID",
			err: &APIError{
				StatusCode: http.StatusInternalServerError,
				Message:    "Internal Server Error",
				RequestID:  "test-req-12345",
			},
		},
		{
			name: "error with URL and method",
			err: &APIError{
				StatusCode: http.StatusNotFound,
				Message:    "Not Found",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that the error string is non-empty
			got := tt.err.Error()
			assert.NotEmpty(t, got, "Error() should return non-empty string")

			// Verify it contains the message
			// We don't mandate exact format, just that the message is present
			if tt.err.Message != "" {
				assert.Contains(t, got, tt.err.Message, "Error() should contain message")
			}

			// If there's a request ID, verify it's included
			if tt.err.RequestID != "" {
				assert.Contains(t, got, tt.err.RequestID, "Error() should contain request ID")
			}
		})
	}
}

func TestAPIError_IsClientError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       bool
	}{
		{"400 is client error", http.StatusBadRequest, true},
		{"404 is client error", http.StatusNotFound, true},
		{"499 is client error", 499, true},
		{"200 is not client error", http.StatusOK, false},
		{"500 is not client error", http.StatusInternalServerError, false},
		{"300 is not client error", http.StatusMultipleChoices, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &APIError{StatusCode: tt.statusCode}
			assert.Equal(t, tt.want, err.IsClientError(), "APIError.IsClientError()")
		})
	}
}

func TestAPIError_IsServerError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       bool
	}{
		{"500 is server error", http.StatusInternalServerError, true},
		{"503 is server error", http.StatusServiceUnavailable, true},
		{"599 is server error", 599, true},
		{"400 is not server error", http.StatusBadRequest, false},
		{"200 is not server error", http.StatusOK, false},
		{"600 is not server error", 600, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &APIError{StatusCode: tt.statusCode}
			assert.Equal(t, tt.want, err.IsServerError(), "APIError.IsServerError()")
		})
	}
}

func TestAPIError_IsAuthenticationError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       bool
	}{
		{"401 is authentication error", http.StatusUnauthorized, true},
		{"400 is not authentication error", http.StatusBadRequest, false},
		{"403 is not authentication error", http.StatusForbidden, false},
		{"500 is not authentication error", http.StatusInternalServerError, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &APIError{StatusCode: tt.statusCode}
			assert.Equal(
				t,
				tt.want,
				err.IsAuthenticationError(),
				"APIError.IsAuthenticationError()",
			)
		})
	}
}

func TestAPIError_IsRateLimitError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       bool
	}{
		{"429 is rate limit error", http.StatusTooManyRequests, true},
		{"400 is not rate limit error", http.StatusBadRequest, false},
		{"500 is not rate limit error", http.StatusInternalServerError, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &APIError{StatusCode: tt.statusCode}
			assert.Equal(t, tt.want, err.IsRateLimitError(), "APIError.IsRateLimitError()")
		})
	}
}

func TestSDKError_Error(t *testing.T) {
	t.Parallel()

	underlying := errors.New("root cause")
	tests := []struct {
		name string
		err  *SDKError
	}{
		{
			name: "message only",
			err: &SDKError{
				Kind:    SDKErrorKindConfiguration,
				Message: "token cannot be empty",
			},
		},
		{
			name: "message with underlying error",
			err: &SDKError{
				Kind:    SDKErrorKindConfiguration,
				Message: "invalid base URL",
				Err:     underlying,
			},
		},
		{
			name: "underlying error only",
			err: &SDKError{
				Kind: SDKErrorKindSerialization,
				Err:  underlying,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.NotEmpty(t, tt.err.Kind, "Kind should not be empty")

			got := tt.err.Error()
			assert.NotEmpty(t, got, "Error() should return non-empty string")
			assert.Contains(t, got, string(tt.err.Kind), "Error() should mention kind")
			if tt.err.Message != "" {
				assert.Contains(t, got, tt.err.Message, "Error() should mention message")
			}
			if tt.err.Err != nil {
				assert.Contains(
					t,
					got,
					tt.err.Err.Error(),
					"Error() should mention underlying error",
				)
				assert.ErrorIs(
					t,
					tt.err,
					tt.err.Err,
					"expected errors.Is to match underlying error",
				)
			}
		})
	}
}
