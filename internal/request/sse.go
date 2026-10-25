package request

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// RawEvent represents a single parsed SSE event payload.
//
// It is an internal detail of the module's SSE wiring: codecs never see it,
// receiving only the curated projection in hfproviders.DecodeParams (the
// frame's payload and event name), while parser state such as ID and Retry
// stays out of the codec contract. Callers wanting raw SSE framing use the
// public hfraw.Event type instead.
type RawEvent struct {
	// Data holds the concatenated data lines for the event.
	Data []byte
	// Event indicates the optional "event:" field name.
	Event string
	// ID is the optional "id:" field value.
	ID string
	// Retry holds the parsed reconnection duration when present.
	Retry *time.Duration
}

// rawResult wraps either an SSE event or an error from the decoder.
// It stays internal so callers only interact with RawEvent via Recv.
type rawResult struct {
	event RawEvent
	err   error
}

// RawStream consumes an SSE response body and exposes parsed events.
type RawStream struct {
	// results receives decoded events/errors from the background goroutine.
	results <-chan rawResult
	// cancel stops the background decoder context.
	cancel context.CancelFunc
	// body is the underlying HTTP response body being consumed.
	body io.ReadCloser
	// bodyOnce ensures the body is closed exactly once.
	bodyOnce sync.Once
	// closeOnce ensures Close() only shuts down the stream a single time.
	closeOnce sync.Once
	// closeError caches the error returned by the underlying body Close.
	closeError error
}

// StreamRaw starts decoding Server-Sent Events from the provided body using
// DefaultMaxResponseBodyBytes as the frame size limit.
// Callers should close the returned RawStream when they are done consuming it so
// the background goroutine and HTTP body are released promptly (otherwise it
// will only stop once the server closes the stream).
func StreamRaw(ctx context.Context, body io.ReadCloser) (*RawStream, error) {
	return StreamRawWithLimit(ctx, body, DefaultMaxResponseBodyBytes)
}

// StreamRawWithLimit starts decoding Server-Sent Events from the provided body.
// maxEventBytes bounds both a single SSE line and the accumulated event data
// (the concatenated data fields). Values <= 0 fall back to
// DefaultMaxResponseBodyBytes.
// Callers should close the returned RawStream when they are done consuming it so
// the background goroutine and HTTP body are released promptly (otherwise it
// will only stop once the server closes the stream).
func StreamRawWithLimit(
	ctx context.Context,
	body io.ReadCloser,
	maxEventBytes int64,
) (*RawStream, error) {
	if body == nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "sse: body is nil",
			Err:     nil,
		}
	}

	maxLineBytes := normalizeSSELimit(maxEventBytes)
	ctx, cancel := context.WithCancel(utils.NormalizeContext(ctx))

	// Buffered with size 1 so the decoder goroutine can enqueue a single event or
	// error without blocking, while still applying backpressure once the caller
	// falls behind (preventing unbounded buffering).
	results := make(chan rawResult, 1)
	stream := &RawStream{
		results:    results,
		cancel:     cancel,
		body:       body,
		bodyOnce:   sync.Once{},
		closeOnce:  sync.Once{},
		closeError: nil,
	}

	go stream.run(ctx, results, maxLineBytes)

	return stream, nil
}

// normalizeSSELimit converts the configured byte limit into the int limit
// required by bufio.Scanner, clamping to the platform's maximum int.
func normalizeSSELimit(maxBytes int64) int {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBodyBytes
	}
	if maxBytes > int64(math.MaxInt) {
		maxBytes = int64(math.MaxInt)
	}

	return int(maxBytes)
}

// Recv blocks until the next event is available, the provided context is canceled,
// or the stream ends. It returns io.EOF when no more events remain.
func (s *RawStream) Recv(ctx context.Context) (event RawEvent, err error) {
	ctx = utils.NormalizeContext(ctx)

	for {
		select {
		case <-ctx.Done():
			return event, ctx.Err()
		case res, ok := <-s.results:
			if !ok {
				return event, io.EOF
			}
			if res.err != nil {
				return event, res.err
			}

			return res.event, nil
		}
	}
}

// Close stops the decoder and releases the underlying body. It is safe to call multiple times.
func (s *RawStream) Close() error {
	s.closeOnce.Do(func() {
		s.cancel()
		s.bodyOnce.Do(func() {
			if err := s.body.Close(); err != nil {
				s.closeError = wrapStreamError(
					err,
					hferrors.SDKErrorKindTransport,
					"close stream body",
				)
			}
		})
	})

	return s.closeError
}

// run drives the background goroutine that parses the SSE stream and forwards
// data or errors onto the results channel. The helper returns as soon as the
// context is canceled, the body ends, or handleLine/emit report that no more
// work should be performed (for example because the send path was canceled).
func (s *RawStream) run(
	ctx context.Context,
	results chan<- rawResult,
	maxLineBytes int,
) {
	defer close(results)
	defer func() { _ = s.Close() }()

	scanner := bufio.NewScanner(s.body)
	scanner.Buffer(nil, maxLineBytes)
	state := &sseState{
		data:         bytes.Buffer{},
		eventType:    "",
		eventID:      "",
		retrySet:     false,
		retry:        0,
		maxDataBytes: int64(maxLineBytes),
		err:          nil,
	}

	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r\n")
		if !state.handleLine(ctx, line, results) {
			if state.err != nil {
				_ = sendResult(ctx, results, rawResult{
					event: RawEvent{
						Data:  nil,
						Event: "",
						ID:    "",
						Retry: nil,
					},
					err: state.err,
				})
			}

			return
		}
	}

	if err := scanner.Err(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			err = ctxErr
		}
		// Any data parsed before the error has already been emitted, so
		// callers just observe the terminal error once they drain the queue.
		if !sendResult(ctx, results, rawResult{
			event: RawEvent{
				Data:  nil,
				Event: "",
				ID:    "",
				Retry: nil,
			},
			err: wrapStreamError(err, hferrors.SDKErrorKindTransport, "read sse line"),
		}) {
			return
		}

		return
	}

	_ = state.emit(ctx, results)
}

func parseSSEField(line string) (field, value string) {
	// parseSSEField splits an SSE line into its field/value components while
	// trimming the optional leading space defined by the spec.
	const sseFieldSplitParts = 2

	parts := strings.SplitN(line, ":", sseFieldSplitParts)
	field = parts[0]
	if len(parts) == 1 {
		return field, ""
	}
	value = parts[1]
	if value != "" && value[0] == ' ' {
		value = value[1:]
	}

	return field, value
}

// wrapStreamError converts stream read errors into SDK errors, preserving caller
// cancellation errors as-is while retaining the calling context message.
func wrapStreamError(err error, kind hferrors.SDKErrorKind, contextMsg string) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return err
	default:
		return &hferrors.SDKError{
			Kind:    kind,
			Message: "sse: failed to " + contextMsg,
			Err:     err,
		}
	}
}

// sseState holds SSE field values between flushes.
type sseState struct {
	data         bytes.Buffer
	eventType    string
	eventID      string
	retrySet     bool
	retry        time.Duration
	maxDataBytes int64
	err          error
}

// handleLine processes a single SSE line and emits an event if needed. It
// returns true when parsing should continue. It returns false when the context
// has been canceled or the event exceeds maxDataBytes; callers should check
// s.err for a terminal error.
func (s *sseState) handleLine(ctx context.Context, line string, results chan<- rawResult) bool {
	switch {
	case line == "":
		return s.emit(ctx, results)
	case strings.HasPrefix(line, ":"):
		// comment, ignore
	default:
		field, value := parseSSEField(line)
		switch field {
		case "data":
			if !s.appendData(value) {
				return false
			}
		case "event":
			s.eventType = value
		case "id":
			s.eventID = value
		case "retry":
			if ms, err := strconv.Atoi(value); err == nil {
				s.retrySet = true
				s.retry = time.Duration(ms) * time.Millisecond
			}
		}
	}

	return true
}

// appendData adds a data field to the current event, enforcing the configured
// event payload limit. It reports false when the event would exceed the limit.
func (s *sseState) appendData(value string) bool {
	if s.maxDataBytes > 0 &&
		int64(s.data.Len())+int64(len(value)) > s.maxDataBytes {
		s.err = &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindTransport,
			Message: "sse: event exceeds maximum size",
			Err:     nil,
		}

		return false
	}

	s.data.WriteString(value)
	s.data.WriteByte('\n')

	return true
}

// emit forwards the buffered event (if any) to the results channel. It returns
// true when the event was delivered (or there was nothing to emit) and false
// when the context was canceled before a receiver accepted the result.
func (s *sseState) emit(ctx context.Context, results chan<- rawResult) bool {
	if s.isEmpty() {
		return true
	}

	payload := s.data.Bytes()
	if len(payload) > 0 && payload[len(payload)-1] == '\n' {
		payload = payload[:len(payload)-1]
	}
	data := append([]byte(nil), payload...)

	var retryPtr *time.Duration
	if s.retrySet {
		d := s.retry
		retryPtr = &d
	}

	if !sendResult(ctx, results, rawResult{
		event: RawEvent{
			Data:  data,
			Event: s.eventType,
			ID:    s.eventID,
			Retry: retryPtr,
		},
		err: nil,
	}) {
		return false
	}

	s.reset()

	return true
}

func (s *sseState) isEmpty() bool {
	return s.data.Len() == 0 && s.eventType == "" && s.eventID == "" && !s.retrySet
}

func (s *sseState) reset() {
	s.data.Reset()
	s.eventType = ""
	s.eventID = ""
	s.retrySet = false
	s.retry = 0
}

// sendResult forwards a decoder result, still attempting a non-blocking send
// after cancellation so errors can reach a waiting receiver. It returns true
// when a receiver consumed the result and false when the context was canceled
// and the channel buffer was full (meaning no caller is waiting anymore).
func sendResult(ctx context.Context, results chan<- rawResult, res rawResult) bool {
	select {
	case <-ctx.Done():
		select {
		case results <- res:
			return true
		default:
			return false
		}
	case results <- res:
		return true
	}
}
