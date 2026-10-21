package hfraw

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// Client sends raw HTTP requests using the configured [hfopts.Options].
//
// It is the deliberate exception to the rest of the SDK, where endpoints are
// exposed as root Client methods: [Client] is the advanced escape hatch for
// endpoints the SDK does not model type-safely.
//
// [Client] is immutable and holds a snapshot of the provided options.
type Client struct {
	opts hfopts.Options
}

// NewClient builds a Client with a snapshot of the provided options.
func NewClient(opts ...hfopts.Option) Client {
	return Client{
		opts: hfopts.NewOptions().With(opts...),
	}
}

// Do performs a raw HTTP request with a byte slice body and applies SDK error interpretation on 4xx/5xx responses.
// The caller must close resp.Body on success.
func (c Client) Do(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return c.DoReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoRaw performs a raw HTTP request with a byte slice body without translating 4xx/5xx responses into SDK errors.
// The caller must close resp.Body on success.
func (c Client) DoRaw(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return c.DoRawReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoReader performs a raw HTTP request with a streaming body and applies SDK error interpretation on 4xx/5xx responses.
// The caller must close resp.Body on success.
func (c Client) DoReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return request.Do(
		c.opts.With(opts...),
		method,
		path,
		requestBody,
	)
}

// DoRawReader performs a raw HTTP request with a streaming body without translating 4xx/5xx responses into SDK errors.
// The caller must close resp.Body on success.
func (c Client) DoRawReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return request.DoRaw(
		c.opts.With(opts...),
		method,
		path,
		requestBody,
	)
}

// Stream performs a raw HTTP request and returns an SSE stream, applying SDK error interpretation on 4xx/5xx responses.
// A 4xx/5xx response becomes an *hferrors.APIError whose Body the caller must close. A response that is not a
// server-sent event stream is reported as an *hferrors.SDKError.
// The status code and headers of the response that opened the stream are available via [Stream.StatusCode] and [Stream.Header].
// Callers should close the returned Stream when finished to promptly release the HTTP connection and decoder goroutine.
func (c Client) Stream(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*Stream, error) {
	return c.StreamReader(bytes.NewReader(requestBody), method, path, opts...)
}

// StreamReader performs a raw HTTP request with a streaming body and returns an SSE stream with SDK error interpretation.
// A 4xx/5xx response becomes an *hferrors.APIError whose Body the caller must close. A response that is not a
// server-sent event stream is reported as an *hferrors.SDKError.
// The status code and headers of the response that opened the stream are available via [Stream.StatusCode] and [Stream.Header].
// Callers should close the returned Stream when finished to promptly release the HTTP connection and decoder goroutine.
func (c Client) StreamReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...hfopts.Option,
) (*Stream, error) {
	resolvedOpts := c.opts.With(opts...)
	resp, err := request.Do(
		resolvedOpts,
		method,
		path,
		requestBody,
	)
	if err != nil {
		return nil, err
	}

	// A response that does not advertise an event stream cannot be consumed as
	// SSE; the decoder would silently yield no events and report io.EOF, losing
	// the response. A 4xx/5xx response has already been translated into an
	// *hferrors.APIError by request.Do, so a non-stream response here is a
	// protocol violation reported as a serialization error with the body
	// drained and closed.
	if err := request.ValidateEventStreamResponseContentType(resp.Header); err != nil {
		request.DrainAndCloseBody(resp.Body, resolvedOpts.MaxResponseBodyBytes)

		return nil, err
	}

	ctx := utils.NormalizeContext(resp.Request.Context())

	rawStream, err := request.StreamRawWithLimit(
		ctx,
		resp.Body,
		resolvedOpts.MaxResponseBodyBytes,
	)
	if err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	return &Stream{
		stream:     rawStream,
		statusCode: resp.StatusCode,
		header:     resp.Header,
	}, nil
}

// Stream exposes a raw SSE stream returned by Client stream methods.
type Stream struct {
	stream     *request.RawStream
	statusCode int
	header     http.Header
}

// StatusCode returns the HTTP status code of the response that opened the
// stream. It returns 0 for a zero-value Stream.
func (s *Stream) StatusCode() int {
	return s.statusCode
}

// Header returns the HTTP response headers of the response that opened the
// stream, or nil for a zero-value Stream. The returned map is owned by the
// SDK and must not be mutated.
func (s *Stream) Header() http.Header {
	return s.header
}

// Recv blocks until the next SSE event is available or the context is done.
func (s *Stream) Recv(ctx context.Context) (event Event, err error) {
	if s.stream == nil {
		return event, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "stream is nil",
			Err:     nil,
		}
	}

	rawEvent, err := s.stream.Recv(ctx)
	if err != nil {
		return event, err
	}

	return Event{
		Data:  append([]byte(nil), rawEvent.Data...),
		Event: rawEvent.Event,
		ID:    rawEvent.ID,
		Retry: rawEvent.Retry,
	}, nil
}

// Close releases the underlying stream resources.
func (s *Stream) Close() error {
	if s.stream == nil {
		return nil
	}

	return s.stream.Close()
}

// Event mirrors the SSE fields returned by raw streams.
type Event struct {
	Data  []byte
	Event string
	ID    string
	Retry *time.Duration
}
