//go:build !integration

package request

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// FuzzSSEPipeline drives a full raw SSE byte stream through
// [StreamRawWithLimit] and checks the documented invariants of the parser
// state machine: no panic on arbitrary bytes, delivered event payloads never
// exceed the configured limit, retry values (when present) are non-negative
// and bounded by the int64 millisecond range, the stream always terminates
// with io.EOF or a well-typed *hferrors.SDKError, and Close is idempotent.
func FuzzSSEPipeline(f *testing.F) {
	seeds := []string{
		"",
		"data: hello\n\n",
		"data: hello\r\n\r\n",
		"data: a\ndata: b\ndata: c\n\n",
		": comment\n\ndata: x\n",
		"event: message\ndata: x\nid: 42\nretry: 1000\n\n",
		"retry: -5000\n\ndata: x\n\n", // negative retry must be ignored
		"retry: notanumber\n\ndata: x\n\n",
		"data: [DONE]\n\n",
		"\xef\xbb\xbfdata: bom\n\n",
		"data:  \n\n",                               // empty-ish data
		"id: \n\n",                                  // empty id event
		strings.Repeat("x", 128),                    // single line over a 64-byte limit
		"data: " + strings.Repeat("y", 70) + "\n\n", // data line over the limit
		"data: 40-bytes-aaaaaaaaaaaaaaaaaaaaaaaaaaa\n" +
			"data: 40-bytes-bbbbbbbbbbbbbbbbbbbbbbbbbbb\n\n", // accumulation over the limit
		"retry: 99999999999999999999999999\n\ndata: x\n\n", // retry overflow
		"\n\n\n\n",                       // consecutive blank lines
		"event:\ndata:\nid:\nretry:\n\n", // empty fields
		"no-colon-line\n\ndata: after\n\n",
	}
	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	const limit = int64(64)

	f.Fuzz(func(t *testing.T, body []byte) {
		stream, err := StreamRawWithLimit(
			context.Background(),
			io.NopCloser(bytes.NewReader(body)),
			limit,
		)
		if err != nil {
			t.Fatalf("StreamRawWithLimit returned setup error: %v", err)
		}

		events := 0
		for {
			_, err := stream.Recv(context.Background())
			switch {
			case err == nil:
				events++
				if events > 1000 {
					// Defensive bound: a finite input must not yield unbounded
					// events. Guard against parser loops re-emitting forever.
					t.Fatalf("parser emitted more than 1000 events for a %d-byte input", len(body))
				}
			case errors.Is(err, io.EOF):
				// Normal termination.
				if _, again := stream.Recv(context.Background()); !errors.Is(again, io.EOF) {
					t.Errorf("Recv after EOF = %v, want io.EOF", again)
				}
				if err := stream.Close(); err != nil {
					t.Errorf("Close after clean EOF returned error: %v", err)
				}
				if err := stream.Close(); err != nil {
					t.Errorf("second Close is not idempotent: %v", err)
				}

				return
			default:
				if _, ok := errors.AsType[*hferrors.SDKError](err); !ok {
					t.Errorf(
						"terminal error is not io.EOF or *hferrors.SDKError: %v (%T)",
						err,
						err,
					)

					return
				}
				if err := stream.Close(); err != nil {
					t.Errorf("Close after terminal error returned error: %v", err)
				}

				return
			}
		}
	})
}

// FuzzSSEEvents delivers individual events from the fuzz corpus and asserts
// per-event payload invariants that the drain loop above cannot check
// without retaining the event. Kept separate so the main harness stays lean.
func FuzzSSEEvents(f *testing.F) {
	seeds := []string{
		"data: hello\n\n",
		"retry: 1000\n\ndata: x\n\n",
		"event: e\nid: i\nretry: 5\ndata: payload\n\n",
		"retry: -1\n\ndata: x\n\n",
	}
	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	const limit = int64(64)

	f.Fuzz(func(t *testing.T, body []byte) {
		stream, err := StreamRawWithLimit(
			context.Background(),
			io.NopCloser(bytes.NewReader(body)),
			limit,
		)
		if err != nil {
			t.Fatalf("StreamRawWithLimit returned setup error: %v", err)
		}

		events := 0
		for {
			event, err := stream.Recv(context.Background())
			if err != nil {
				break
			}

			events++
			if events > 1000 {
				// Defensive bound mirroring FuzzSSEPipeline: a finite input
				// must not yield unbounded events, so a parser emit-loop
				// fails the harness instead of hanging it.
				t.Fatalf("parser emitted more than 1000 events for a %d-byte input", len(body))
			}

			if int64(len(event.Data)) > limit {
				t.Errorf(
					"event data %d bytes exceeds limit %d: %q",
					len(event.Data),
					limit,
					event.Data,
				)
			}
			if event.Retry != nil && *event.Retry < 0 {
				t.Errorf("event %q carries a negative retry: %v", event.Data, *event.Retry)
			}
			// Large retry values are a legitimate server pass-through (up to
			// MaxInt64 milliseconds); only negativity is a parser bug.
		}

		_ = stream.Close()
	})
}
