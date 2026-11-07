//go:build !integration

package utils

import (
	"net/http"
	"testing"
	"time"
)

// FuzzParseRetryAfter checks the documented invariants of
// [ParseRetryAfter]: the result is never negative and never exceeds the
// configured limit, regardless of the header value. In particular, a header
// value large enough to overflow the seconds-to-duration multiplication must
// not wrap into a negative (or bogus positive) delay.
func FuzzParseRetryAfter(f *testing.F) {
	seeds := []string{
		"",
		" ",
		"1",
		"0",
		"-1",
		"60",
		"61",
		"120",
		"abc",
		"1.5",
		" 42 ",
		"+42",
		"9223372036854775807",          // max int64
		"9223372036",                   // just below int64(time.Second) overflow threshold
		"9223372037",                   // at the overflow threshold
		"999999999999999999",           // overflows Atoi on 64-bit
		"18446744073709551617",         // overflows Atoi on all platforms
		string([]byte{'9', '1', '\t'}), // odd whitespace variant
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, value string) {
		limit := 60 * time.Second
		h := http.Header{}
		h.Set("Retry-After", value)

		got := ParseRetryAfter(h, limit)
		if got < 0 {
			t.Errorf("ParseRetryAfter(%q) = %v, want a non-negative duration", value, got)
		}
		if got > limit {
			t.Errorf("ParseRetryAfter(%q) = %v, want at most %v", value, got, limit)
		}

		// Zero limit must yield zero: any positive delay would exceed it.
		if zero := ParseRetryAfter(h, 0); zero != 0 {
			t.Errorf("ParseRetryAfter(%q) with limit 0 = %v, want 0", value, zero)
		}
	})
}
