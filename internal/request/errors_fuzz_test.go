//go:build !integration

package request

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// messageSuffix is the truncation marker appended by [messageBody].
const messageSuffix = "... (truncated)"

// FuzzAPIErrorMessage checks the documented invariants of [messageBody],
// the sanitizer behind [hferrors.APIError.Message]: the rendered message
// never contains control bytes (so server-controlled bodies cannot forge log
// lines), respects the 4 KiB cap plus marker, appends the truncation marker
// exactly when the output was cut short, keeps valid UTF-8 input valid after
// rune-boundary backoff, and leaves short untruncated input byte-identical
// (modulo control-byte stripping).
func FuzzAPIErrorMessage(f *testing.F) {
	seeds := []string{
		"",
		"plain error body",
		"line1\nline2\rtab\tend",
		"\x00\x01\x02\x03",
		"\x7f",
		"héllo wörld",
		strings.Repeat("a", 4096),
		strings.Repeat("a", 4097),
		strings.Repeat("ab", 4096) + "×", // multi-byte rune at the cap boundary
		"\xff\xfe invalid utf8 \x80",
		"a" + strings.Repeat("é", 3000), // multi-byte overflow of the cap
		"\xed\xa0\x80",                  // surrogate half (invalid UTF-8)
		"é" + string([]byte{0xC3}),      // truncated multi-byte tail
	}
	for _, seed := range seeds {
		f.Add([]byte(seed), false)
		f.Add([]byte(seed), true)
	}

	f.Fuzz(func(t *testing.T, body []byte, truncated bool) {
		got := messageBody(body, truncated)

		for i, b := range got {
			if b < 0x20 || b == 0x7f {
				t.Fatalf(
					"messageBody(%q, %v) = %q contains control byte %x at %d",
					body,
					truncated,
					got,
					b,
					i,
				)
			}
		}

		if maxLen := apiErrorMessageCapBytes + len(messageSuffix); len(got) > maxLen {
			t.Errorf(
				"messageBody(%q, %v) = %q length %d exceeds cap %d",
				body,
				truncated,
				got,
				len(got),
				maxLen,
			)
		}

		// The marker is appended whenever the rendered portion was cut
		// short: by the read-time truncation flag or by the 4 KiB cap on
		// the control-stripped copy.
		expectTruncated := truncated ||
			int64(len(stripControlBytes(body))) > apiErrorMessageCapBytes
		hasSuffix := strings.HasSuffix(got, messageSuffix)
		if expectTruncated && !hasSuffix {
			t.Errorf("messageBody(%q, %v) = %q missing truncation marker", body, truncated, got)
		}
		if !expectTruncated && hasSuffix {
			t.Errorf("messageBody(%q, %v) = %q unexpectedly truncated", body, truncated, got)
		}

		core := strings.TrimSuffix(got, messageSuffix)
		if utf8.Valid(body) && !utf8.ValidString(core) {
			t.Errorf(
				"messageBody(%q, %v) = %q produced invalid UTF-8 from valid input",
				body,
				truncated,
				core,
			)
		}

		if !expectTruncated {
			if want := string(stripControlBytes(body)); got != want {
				t.Errorf("messageBody(%q, %v) = %q, want %q", body, truncated, got, want)
			}
		}
	})
}
