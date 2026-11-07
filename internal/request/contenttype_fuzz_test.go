//go:build !integration

package request

import (
	"mime"
	"net/http"
	"testing"
)

// FuzzEventStreamContentType checks the SSE response Content-Type guard:
// [ValidateEventStreamResponseContentType] never panics on arbitrary header
// values, non-event-stream media types are rejected, and a passing value
// parses to exactly text/event-stream.
func FuzzEventStreamContentType(f *testing.F) {
	seeds := []string{
		"",
		"text/event-stream",
		"text/event-stream; charset=utf-8",
		"TEXT/EVENT-STREAM",
		"application/json",
		"text/plain",
		"garbage",
		"/",
		";",
		"text/event-stream;;;",
		"text/event-stream; charset=",
		"text/event-stream; =x",
		"text/event-stream;boundary=\x00",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, contentType string) {
		headers := http.Header{}
		if contentType != "" {
			headers.Set("Content-Type", contentType)
		}

		err := ValidateEventStreamResponseContentType(headers)

		// The guard defers to mime.ParseMediaType: a value is accepted iff
		// it parses to exactly text/event-stream (case-insensitive, media
		// type parameters such as charset allowed).
		mediaType, _, parseErr := mime.ParseMediaType(contentType)
		accepted := parseErr == nil && mediaType == "text/event-stream"
		if contentType == "" {
			accepted = false // empty header must always be rejected
		}

		if (err == nil) != accepted {
			t.Errorf(
				"ValidateEventStreamResponseContentType(%q) = %v, want acceptance=%v (mime parses to %q, err=%v)",
				contentType,
				err,
				accepted,
				mediaType,
				parseErr,
			)
		}
	})
}
