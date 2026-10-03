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

// Client sends raw HTTP requests using the configured request options.
//
// It is the deliberate exception to the rest of the SDK, where endpoints are
// exposed as root Client methods: Client is the advanced escape hatch for
// endpoints the SDK does not model type-safely. It combines byte-slice or
// io.Reader bodies, typed error handling or raw HTTP responses, and one-shot
// or SSE streaming into eight methods.
//
// Client is immutable and holds a snapshot of the provided options.
type Client struct {
	opts hfopts.Options
}

// NewClient builds a Client with a snapshot of the provided options.
func NewClient(opts ...hfopts.Option) Client {
	return Client{
		opts: hfopts.NewOptions().With(opts...),
	}
}

// Do performs a raw HTTP request with a byte slice body and applies SDK error interpretation on non-2xx responses.
// The caller must close resp.Body on success.
func (c Client) Do(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return c.DoReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoRaw performs a raw HTTP request with a byte slice body without translating non-2xx responses into SDK errors.
// The caller must close resp.Body on success.
func (c Client) DoRaw(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*http.Response, error) {
	return c.DoRawReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoReader performs a raw HTTP request with a streaming body and applies SDK error interpretation on non-2xx responses.
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

// DoRawReader performs a raw HTTP request with a streaming body without translating non-2xx responses into SDK errors.
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

// Stream performs a raw HTTP request and returns an SSE stream, applying SDK error interpretation on non-2xx responses.
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
// Callers should close the returned Stream when finished to promptly release the HTTP connection and decoder goroutine.
func (c Client) StreamReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...hfopts.Option,
) (*Stream, error) {
	return c.doStream(requestBody, method, path, opts, false)
}

// StreamRaw performs a raw HTTP request and returns an SSE stream without translating non-2xx responses into SDK errors.
// This function is probably only interesting to advanced users.
// Only use this when you need to inspect the raw response; callers are responsible for interpreting HTTP errors themselves.
// Callers should close the returned Stream when finished to promptly release the HTTP connection and decoder goroutine.
func (c Client) StreamRaw(
	requestBody []byte,
	method string,
	path string,
	opts ...hfopts.Option,
) (*Stream, error) {
	return c.StreamRawReader(bytes.NewReader(requestBody), method, path, opts...)
}

// StreamRawReader performs a raw HTTP request with a streaming body and returns an SSE stream without translating non-2xx responses into SDK errors.
// This function is probably only interesting to advanced users.
// Only use this when you need to inspect the raw response; callers are responsible for interpreting HTTP errors themselves.
// Callers should close the returned Stream when finished to promptly release the HTTP connection and decoder goroutine.
func (c Client) StreamRawReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...hfopts.Option,
) (*Stream, error) {
	return c.doStream(requestBody, method, path, opts, true)
}

// doStream performs the underlying HTTP request and SSE stream setup,
// selecting between SDK-error-translating (Do) and raw (DoRaw) execution.
func (c Client) doStream(
	requestBody io.Reader,
	method string,
	path string,
	opts []hfopts.Option,
	raw bool,
) (*Stream, error) {
	var resp *http.Response
	var err error

	if raw {
		resp, err = request.DoRaw(
			c.opts.With(opts...),
			method,
			path,
			requestBody,
		)
	} else {
		resp, err = request.Do(
			c.opts.With(opts...),
			method,
			path,
			requestBody,
		)
	}
	if err != nil {
		return nil, err
	}

	ctx := utils.NormalizeContext(resp.Request.Context())

	rawStream, err := request.StreamRaw(ctx, resp.Body)
	if err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	return &Stream{stream: rawStream}, nil
}

// Stream exposes a raw SSE stream returned by Client stream methods.
type Stream struct {
	stream *request.RawStream
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
