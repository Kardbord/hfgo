package testutils

import "encoding/base64"

// TestAudioWAV is a valid WAV audio file used for integration tests.
// It is a 1-second sine wave tone encoded as a base64 string.
const TestAudioWAV = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="

// DecodeTestAudioWAV decodes TestAudioWAV to raw bytes.
func DecodeTestAudioWAV() []byte {
	b, _ := base64.StdEncoding.DecodeString(TestAudioWAV)

	return b
}
