package request

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
)

const (
	mimeApplicationJSON = "application/json"
	mimeEventStream     = "text/event-stream"
)

// DecodeHTTPResponse handles the common HTTP response decoding logic:
// 204/205 status codes, content-type validation, body reading, and empty-body
// detection. It returns nil body for 204/205 (the caller should return a zero
// value in that case).
func DecodeHTTPResponse(resp *http.Response, maxResponseBodyBytes int64) (body []byte, err error) {
	if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusResetContent {
		return nil, nil
	}
	if err := ValidateJSONResponseContentType(resp.Header); err != nil {
		return nil, err
	}

	body, err = ReadResponseBody(resp, maxResponseBodyBytes)
	if err != nil {
		return nil, err
	}
	if len(body) == 0 {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "empty response body",
			Err:     nil,
		}
	}

	return body, nil
}

// UnmarshalJSONResponse unmarshals a JSON response body into the target type.
// It maps io.EOF to an "empty response body" SDK error.
func UnmarshalJSONResponse[T any](body []byte, target *T) error {
	if err := json.Unmarshal(body, target); err != nil {
		if errors.Is(err, io.EOF) {
			return &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "empty response body",
				Err:     err,
			}
		}

		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode response body",
			Err:     err,
		}
	}

	return nil
}

// JSONStream consumes JSON SSE events.
type JSONStream[T any] struct {
	raw *RawStream
	// decode is an optional transform applied to each data chunk before
	// unmarshalling. If nil, the chunk is used as-is.
	decode func([]byte) ([]byte, error)
}

// NewJSONStream returns a JSONStream backed by the given RawStream.
// If decode is non-nil, it is applied to each data chunk before
// unmarshalling into T.
func NewJSONStream[T any](raw *RawStream, decode func([]byte) ([]byte, error)) *JSONStream[T] {
	return &JSONStream[T]{raw: raw, decode: decode}
}

// Recv blocks until the next JSON event is available or the stream ends.
// It skips keepalive events, treats data: [DONE] as EOF, and unmarshals each chunk into T.
func (s *JSONStream[T]) Recv(ctx context.Context) (out T, err error) {
	if s.raw == nil {
		return out, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "json stream is nil",
			Err:     nil,
		}
	}

	for {
		event, err := s.raw.Recv(ctx)
		if err != nil {
			return out, err
		}
		data := bytes.TrimSpace(event.Data)
		if len(data) == 0 {
			continue
		}
		if bytes.Equal(data, []byte("[DONE]")) {
			_ = s.raw.Close()

			return out, io.EOF
		}
		if s.decode != nil {
			data, err = s.decode(data)
			if err != nil {
				return out, &hferrors.SDKError{
					Kind:    hferrors.SDKErrorKindSerialization,
					Message: "failed to decode stream event",
					Err:     err,
				}
			}
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return out, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "failed to decode stream event",
				Err:     err,
			}
		}

		return out, nil
	}
}

// Close releases the underlying stream resources.
func (s *JSONStream[T]) Close() error {
	if s.raw == nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "json stream is nil",
			Err:     nil,
		}
	}

	return s.raw.Close()
}

// ensureHeader returns a copy of headers with a default value set when missing or empty.
func ensureHeader(h http.Header, key, value string) http.Header {
	out := cloneHeader(h)
	if out == nil {
		out = make(http.Header, 1)
	}
	if v := out.Get(key); v == "" {
		out.Set(key, value)
	}

	return out
}

// ValidateJSONRequestContentType validates that Content-Type is application/json when provided.
func ValidateJSONRequestContentType(headers http.Header) error {
	contentType := headers.Get("Content-Type")
	if contentType == "" {
		return nil
	}
	mediatype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "invalid Content-Type header",
			Err:     err,
		}
	}
	if mediatype != mimeApplicationJSON {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "Content-Type must be application/json for DoJSON requests",
			Err:     nil,
		}
	}

	return nil
}

// ValidateJSONResponseContentType validates that the response Content-Type indicates JSON.
func ValidateJSONResponseContentType(headers http.Header) error {
	contentType := headers.Get("Content-Type")
	if contentType == "" {
		return nil
	}
	mediatype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "invalid Content-Type header on response",
			Err:     err,
		}
	}
	if !isJSONMediaType(mediatype) {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "response Content-Type must be application/json",
			Err:     nil,
		}
	}

	return nil
}

// ValidateEventStreamResponseContentType ensures the response advertises text/event-stream.
func ValidateEventStreamResponseContentType(headers http.Header) error {
	contentType := headers.Get("Content-Type")
	if contentType == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "response Content-Type must be text/event-stream",
			Err:     nil,
		}
	}
	mediatype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "invalid Content-Type header on response",
			Err:     err,
		}
	}
	if mediatype != mimeEventStream {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "response Content-Type must be text/event-stream",
			Err:     nil,
		}
	}

	return nil
}

// isJSONMediaType reports whether the media type is JSON or a +json subtype.
func isJSONMediaType(mediatype string) bool {
	if mediatype == mimeApplicationJSON {
		return true
	}

	return strings.HasSuffix(mediatype, "+json")
}
