package testutils

import (
	"errors"
	"io"
)

// CloseTracker tracks whether Close is called.
type CloseTracker struct {
	Closed bool
}

// Read implements io.Reader and always returns io.EOF.
func (c *CloseTracker) Read([]byte) (int, error) {
	return 0, io.EOF
}

// Close marks the tracker as closed.
func (c *CloseTracker) Close() error {
	c.Closed = true

	return nil
}

// ReadTracker captures bytes read and closed state for a backing byte slice.
type ReadTracker struct {
	Data      []byte
	Offset    int
	ReadBytes int
	Closed    bool
}

// Read implements io.Reader and tracks bytes read.
func (r *ReadTracker) Read(p []byte) (int, error) {
	if r.Offset >= len(r.Data) {
		return 0, io.EOF
	}
	n := copy(p, r.Data[r.Offset:])
	r.Offset += n
	r.ReadBytes += n

	return n, nil
}

// Close marks the tracker as closed.
func (r *ReadTracker) Close() error {
	r.Closed = true

	return nil
}

// ErrorReadCloser is an io.ReadCloser that always returns an error.
type ErrorReadCloser struct{}

// Read implements io.Reader and always returns an error.
func (e ErrorReadCloser) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}

// Close implements io.Closer.
func (e ErrorReadCloser) Close() error {
	return nil
}
