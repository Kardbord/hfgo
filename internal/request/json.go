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

// DoJSON performs an HTTP request with a JSON request body and expects a JSON response.
// It marshals the request body to JSON, sends the request, and unmarshals the response
// into the specified response type. The function uses Go generics to provide type-safe
// request and response handling.
//
// Type parameters:
//   - TReq: The type of the request body
//   - TResp: The type of the response body
//
// Returns an error if JSON marshaling/unmarshaling fails or the HTTP request fails.
// For HTTP errors, Do returns an *errors.APIError which includes the status code,
// response body, and other metadata.
//
//nolint:bodyclose // DrainAndCloseBody closes the response body.
func DoJSON[TReq any, TResp any](
	opts Options,
	method string,
	path string,
	reqBody TReq,
) (resp TResp, err error) {
	buf, err := marshalJSONRequestBody(reqBody)
	if err != nil {
		return resp, err
	}

	opts, err = prepareJSONOptions(opts, mimeApplicationJSON)
	if err != nil {
		return resp, err
	}

	httpResp, err := DoBytes(opts, method, path, buf)
	if err != nil {
		return resp, err
	}
	defer DrainAndCloseBody(httpResp.Body)

	resp, err = decodeJSONResponse[TResp](httpResp, opts.MaxResponseBodyBytes)

	return resp, err
}

func decodeJSONResponse[T any](resp *http.Response, maxResponseBodyBytes int64) (out T, err error) {
	if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusResetContent {
		return out, nil
	}
	if err := ValidateJSONResponseContentType(resp.Header); err != nil {
		return out, err
	}

	body, err := ReadResponseBody(resp, maxResponseBodyBytes)
	if err != nil {
		return out, err
	}
	if len(body) == 0 {
		return out, &hferrors.SDKError{
			Kind: hferrors.SDKErrorKindSerialization,

			Message: "empty response body",
			Err:     nil,
		}
	}

	if err := json.Unmarshal(body, &out); err != nil {
		if errors.Is(err, io.EOF) {
			return out, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: "empty response body",
				Err:     err,
			}
		}

		return out, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to decode response body",
			Err:     err,
		}
	}

	return out, nil
}

// DoJSONStream performs an HTTP request with a JSON body and returns a streaming JSON response.
// The response body must be a Server-Sent Events (SSE) stream where each data chunk contains JSON.
// Callers are responsible for closing the returned stream to release resources.
func DoJSONStream[TReq any, TResp any](
	opts Options,
	method string,
	path string,
	reqBody TReq,
) (*JSONStream[TResp], error) {
	buf, err := marshalJSONRequestBody(reqBody)
	if err != nil {
		return nil, err
	}

	opts, err = prepareJSONOptions(opts, mimeEventStream)
	if err != nil {
		return nil, err
	}

	resp, err := DoBytes(
		opts,
		method,
		path,
		buf,
	)
	if err != nil {
		return nil, err
	}

	if err := ValidateEventStreamResponseContentType(resp.Header); err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	raw, err := StreamRaw(opts.Context(), resp.Body)
	if err != nil {
		_ = resp.Body.Close()

		return nil, err
	}

	return NewJSONStream[TResp](raw, nil), nil
}

// JSONStream consumes JSON SSE events produced by DoJSONStream.
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

// validateJSONRequestContentType validates that Content-Type is application/json when provided.
func validateJSONRequestContentType(headers http.Header) error {
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

// marshalJSONRequestBody serializes the payload and normalizes errors to SDK errors.
func marshalJSONRequestBody(payload any) ([]byte, error) {
	buf, err := json.Marshal(payload)
	if err != nil {
		var sdkErr *hferrors.SDKError
		if errors.As(err, &sdkErr) {
			return nil, sdkErr
		}

		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to marshal request body",
			Err:     err,
		}
	}

	return buf, nil
}

// prepareJSONOptions sets standard headers and validates Content-Type for JSON requests.
func prepareJSONOptions(opts Options, accept string) (Options, error) {
	opts = opts.WithDefaultHeader("Content-Type", mimeApplicationJSON)
	opts = opts.WithDefaultHeader("Accept", accept)
	if err := validateJSONRequestContentType(opts.Headers); err != nil {
		return Options{}, err
	}

	return opts, nil
}

// isJSONMediaType reports whether the media type is JSON or a +json subtype.
func isJSONMediaType(mediatype string) bool {
	if mediatype == mimeApplicationJSON {
		return true
	}

	return strings.HasSuffix(mediatype, "+json")
}
