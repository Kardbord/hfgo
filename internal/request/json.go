package request

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

const mimeEventStream = "text/event-stream"

// DecodeHTTPResponse handles the common HTTP response decoding logic:
// 204/205 status codes, body reading, and empty-body detection. It is
// format-agnostic: response content-type validation is owned by the codec
// that decodes the body. It returns a nil body for 204/205 (the caller should
// return a zero value in that case).
func DecodeHTTPResponse(resp *http.Response, maxResponseBodyBytes int64) (body []byte, err error) {
	if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusResetContent {
		return nil, nil
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

// JSONStream consumes SSE events, delivering each event payload as a value of
// type T.
type JSONStream[T any] struct {
	raw *RawStream
	// decode is an optional transform that produces a value from each raw
	// event payload. If nil, the payload is JSON-unmarshalled into T.
	decode func(context.Context, []byte) (T, error)
}

// NewJSONStream returns a JSONStream backed by the given RawStream.
// If decode is non-nil, it produces each value from the raw event payload;
// otherwise the payload is JSON-unmarshalled into T.
func NewJSONStream[T any](
	raw *RawStream,
	decode func(context.Context, []byte) (T, error),
) *JSONStream[T] {
	return &JSONStream[T]{raw: raw, decode: decode}
}

// Recv blocks until the next event is available or the stream ends.
// It skips keepalive events, treats data: [DONE] as EOF, and decodes each
// payload into T.
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
			// Normalize like RawStream.Recv so decode hooks always receive a
			// non-nil context.
			out, err = s.decode(utils.NormalizeContext(ctx), data)
			if err != nil {
				return out, &hferrors.SDKError{
					Kind:    hferrors.SDKErrorKindSerialization,
					Message: "failed to decode stream event",
					Err:     err,
				}
			}

			return out, nil
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
