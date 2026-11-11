package testutils

import (
	"bytes"
	"io"
	"net/http"
)

// MockTransport is a mock implementation of http.RoundTripper for testing purposes.
// It captures the last request made and returns a predefined response and error.
type MockTransport struct {
	LastRequest *http.Request
	Response    *http.Response
	Err         error
}

// NewMockTransport creates a new mock transport with the specified response status,
// body, and error. This is useful for testing HTTP request handling without making
// actual network calls.
func NewMockTransport(respStatus int, respBody string, err error) *MockTransport {
	return &MockTransport{
		Response: &http.Response{
			StatusCode: respStatus,
			Body:       io.NopCloser(bytes.NewBufferString(respBody)),
			Header:     make(http.Header),
		},
		Err: err,
	}
}

// NewJSONMockTransport creates a mock transport with an application/json content type.
func NewJSONMockTransport(respStatus int, respBody string, err error) *MockTransport {
	mt := NewMockTransport(respStatus, respBody, err)
	mt.Response.Header.Set("Content-Type", "application/json")

	return mt
}

// RoundTrip executes a mock HTTP request, storing the request for inspection and returning
// the predefined response and error. It respects context cancellation.
func (m *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	m.LastRequest = req

	select {
	case <-req.Context().Done():
		return nil, req.Context().Err()
	default:
	}

	return m.Response, m.Err
}

// NewMockHTTPClient returns an http.Client that uses the provided mock transport.
func NewMockHTTPClient(mt *MockTransport) http.Client {
	return http.Client{Transport: mt}
}
