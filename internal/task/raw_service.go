package task

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"time"

	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// RawService sends raw HTTP requests using the configured request options.
//
// It is the deliberate exception to the rest of the SDK, where endpoints are
// exposed as Client methods: RawService is the advanced escape hatch for
// endpoints the SDK does not model type-safely. It combines several axes
// (byte-slice or io.Reader bodies, typed error handling or raw HTTP responses,
// one-shot or SSE streaming) into eight methods, which is enough surface that
// keeping it namespaced under Client.Raw() avoids cluttering the Client API.
//
// RawService is immutable and holds a snapshot of the client options.
type RawService struct {
	opts request.Options
}

// NewRawService builds a raw service with a snapshot of the provided options.
func NewRawService(opts request.Options) RawService {
	return RawService{opts: opts}
}

// Do performs a raw HTTP request with a byte slice body and applies SDK error interpretation on non-2xx responses.
// The caller must close resp.Body on success.
func (r RawService) Do(
	requestBody []byte,
	method string,
	path string,
	opts ...request.Option,
) (*http.Response, error) {
	return r.DoReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoRaw performs a raw HTTP request with a byte slice body without translating non-2xx responses into SDK errors.
// The caller must close resp.Body on success.
func (r RawService) DoRaw(
	requestBody []byte,
	method string,
	path string,
	opts ...request.Option,
) (*http.Response, error) {
	return r.DoRawReader(bytes.NewReader(requestBody), method, path, opts...)
}

// DoReader performs a raw HTTP request with a streaming body and applies SDK error interpretation on non-2xx responses.
// The caller must close resp.Body on success.
func (r RawService) DoReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...request.Option,
) (*http.Response, error) {
	return request.Do(
		r.opts.With(opts...),
		method,
		path,
		requestBody,
	)
}

// DoRawReader performs a raw HTTP request with a streaming body without translating non-2xx responses into SDK errors.
// The caller must close resp.Body on success.
func (r RawService) DoRawReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...request.Option,
) (*http.Response, error) {
	return request.DoRaw(
		r.opts.With(opts...),
		method,
		path,
		requestBody,
	)
}

// Stream performs a raw HTTP request and returns an SSE stream, applying SDK error interpretation on non-2xx responses.
// Callers should Close the returned RawStream when finished to promptly release the HTTP connection and decoder goroutine.
func (r RawService) Stream(
	requestBody []byte,
	method string,
	path string,
	opts ...request.Option,
) (*RawStream, error) {
	return r.StreamReader(bytes.NewReader(requestBody), method, path, opts...)
}

// StreamReader performs a raw HTTP request with a streaming body and returns an SSE stream with SDK error interpretation.
// Callers should Close the returned RawStream when finished to promptly release the HTTP connection and decoder goroutine.
func (r RawService) StreamReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...request.Option,
) (*RawStream, error) {
	resp, err := request.Do(
		r.opts.With(opts...),
		method,
		path,
		requestBody,
	)
	if err != nil {
		return nil, err
	}

	ctx := request.NormalizeContext(resp.Request.Context())

	raw, err := request.StreamRaw(ctx, resp.Body)
	if err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	return &RawStream{stream: raw}, nil
}

// StreamRaw performs a raw HTTP request and returns an SSE stream without translating non-2xx responses into SDK errors.
// This function is probably only interesting to advanced users.
// Only use this when you need to inspect the raw response; callers are responsible for interpreting HTTP errors themselves.
// Callers should Close the returned RawStream when finished to promptly release the HTTP connection and decoder goroutine.
func (r RawService) StreamRaw(
	requestBody []byte,
	method string,
	path string,
	opts ...request.Option,
) (*RawStream, error) {
	return r.StreamRawReader(bytes.NewReader(requestBody), method, path, opts...)
}

// StreamRawReader performs a raw HTTP request with a streaming body and returns an SSE stream without translating non-2xx responses into SDK errors.
// This function is probably only interesting to advanced users.
// Only use this when you need to inspect the raw response; callers are responsible for interpreting HTTP errors themselves.
// Callers should Close the returned RawStream when finished to promptly release the HTTP connection and decoder goroutine.
func (r RawService) StreamRawReader(
	requestBody io.Reader,
	method string,
	path string,
	opts ...request.Option,
) (*RawStream, error) {
	resp, err := request.DoRaw(
		r.opts.With(opts...),
		method,
		path,
		requestBody,
	)
	if err != nil {
		return nil, err
	}

	ctx := request.NormalizeContext(resp.Request.Context())

	raw, err := request.StreamRaw(ctx, resp.Body)
	if err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	return &RawStream{stream: raw}, nil
}

// RawStream exposes a raw SSE stream returned by RawService stream methods.
type RawStream struct {
	stream *request.RawStream
}

// Recv blocks until the next SSE event is available or the context is done.
func (s *RawStream) Recv(ctx context.Context) (event RawEvent, err error) {
	if s.stream == nil {
		return event, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindInternal,
			Message: "raw stream is nil",
			Err:     nil,
		}
	}

	rawEvent, err := s.stream.Recv(ctx)
	if err != nil {
		return event, err
	}

	return RawEvent{
		Data:  append([]byte(nil), rawEvent.Data...),
		Event: rawEvent.Event,
		ID:    rawEvent.ID,
		Retry: rawEvent.Retry,
	}, nil
}

// Close releases the underlying stream resources.
func (s *RawStream) Close() error {
	if s.stream == nil {
		return nil
	}

	return s.stream.Close()
}

// RawEvent mirrors the SSE fields returned by raw streams.
type RawEvent struct {
	Data  []byte
	Event string
	ID    string
	Retry *time.Duration
}
