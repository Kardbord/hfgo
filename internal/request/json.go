package request

import (
	"bytes"
	"context"
	"errors"
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
	// decode produces a value from each raw event. It is required and must
	// not be nil.
	decode func(context.Context, RawEvent) (T, error)
}

// NewJSONStream returns a JSONStream backed by the given RawStream.
// decode transforms each raw event into a value of T; it must not be nil
// (callers wanting plain JSON payloads supply a json.Unmarshal-based hook).
func NewJSONStream[T any](
	raw *RawStream,
	decode func(context.Context, RawEvent) (T, error),
) *JSONStream[T] {
	return &JSONStream[T]{raw: raw, decode: decode}
}

// classifyHookError maps a decode-hook error onto stream control behavior.
func classifyHookError(err error) (skip, terminate bool) {
	switch {
	case errors.Is(err, hferrors.SkipEventError{}):
		return true, false
	case errors.Is(err, hferrors.EndOfStreamError{}):
		return false, true
	default:
		return false, false
	}
}

// endStream releases the raw stream and drains any buffered frames so the
// results channel ends closed and empty: the channel itself records
// termination, with no JSONStream state, and every later receive yields
// io.EOF. The drain cannot block unboundedly — Close cancels the producer's
// context and closes the body, so the producer returns promptly and closes
// the channel. Frames the server sent after the terminating frame are
// discarded here and never delivered. The drain deliberately uses a
// non-cancelable context: a caller-canceled ctx could stop the drain while
// a frame is still buffered, breaking the post-EOF terminal guarantee.
func (s *JSONStream[T]) endStream() error {
	_ = s.raw.Close()

	for {
		if _, err := s.raw.Recv(context.Background()); err != nil {
			return io.EOF
		}
	}
}

// Recv blocks until the next event is available or the stream ends.
// It skips keepalive events, treats data: [DONE] as EOF, and decodes each
// payload into T. A decode hook may steer the stream by returning
// [hferrors.SkipEventError] to suppress the current frame or
// [hferrors.EndOfStreamError] to end it (Recv then returns io.EOF).
//
// Termination is final: the terminating frame's own value is not surfaced,
// frames after it are discarded, and anything a consumer receives precedes
// the terminal frame. Recv is intended for a single consumer goroutine;
// concurrent consumers are memory-safe (events partition over a channel)
// but which goroutine receives which frame is undefined.
func (s *JSONStream[T]) Recv(ctx context.Context) (out T, err error) {
	if s.raw == nil {
		return out, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "json stream is nil",
			Err:     nil,
		}
	}

	if s.decode == nil {
		return out, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "json stream decode is nil",
			Err:     nil,
		}
	}

	for {
		event, err := s.raw.Recv(ctx)
		if err != nil {
			return out, err
		}

		next, skip, nextErr := s.processEvent(ctx, event)
		if !skip {
			return next, nextErr
		}
	}
}

// processEvent decodes one raw event and reports the outcome for it.
// skip is true only when the frame produced nothing to return — a keepalive
// or a frame suppressed by a [hferrors.SkipEventError] signal — in which
// case the caller fetches the next event; otherwise (out, err) is the
// result, which may be a delivered value, a failure, or io.EOF at
// termination.
func (s *JSONStream[T]) processEvent(
	ctx context.Context,
	event RawEvent,
) (out T, skip bool, err error) {
	data := bytes.TrimSpace(event.Data)
	if len(data) == 0 {
		return out, true, nil
	}

	if bytes.Equal(data, []byte("[DONE]")) {
		return out, false, s.endStream() //nolint:contextcheck // drain uses its own bounded ctx
	}

	// Normalize like RawStream.Recv so decode hooks always receive a
	// non-nil context.
	out, err = s.decode(utils.NormalizeContext(ctx), event)
	if err == nil {
		return out, false, nil
	}

	hookSkip, hookTerminate := classifyHookError(err)
	switch {
	case hookSkip:
		return out, true, nil
	case hookTerminate:
		var zero T

		return zero, false, s.endStream() //nolint:contextcheck // drain uses its own bounded ctx
	default:
		return out, false, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode stream event",
			Err:     err,
		}
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
