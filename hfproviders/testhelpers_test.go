//go:build !integration

package hfproviders

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// ctJSON is spelled to avoid a JSON-suffixed identifier so testifylint's
// encoded-compare heuristic treats header assertions as plain string
// comparisons.
const ctJSON = "application/json"

// jsonHeaders builds a header set whose Content-Type is the given media type —
// the response shape most codec Decode tests need.
func jsonHeaders(ct string) http.Header {
	return http.Header{"Content-Type": {ct}}
}

// postSubmitParams builds transport parameters targeting the given test
// server, shared by every queue-based provider's PostSubmit exchange tests.
func postSubmitParams(server *httptest.Server) PostSubmitParams {
	return PostSubmitParams{
		HTTPClient:           server.Client(),
		BaseURL:              server.URL,
		Token:                "test-token",
		UserAgent:            "test-agent/1.0",
		MaxResponseBodyBytes: 1 << 20,
	}
}

// recordingResolver records the ResolveModel lookup key and call count and
// serves a fixed mapping, letting tests assert what was looked up and how
// often.
type recordingResolver struct {
	got     ResolveModelParams
	calls   atomic.Int64
	mapping ModelMapping
}

// ResolveModel implements [ModelResolver].
func (r *recordingResolver) ResolveModel(
	_ context.Context,
	params ResolveModelParams,
) (ModelMapping, error) {
	r.got = params
	r.calls.Add(1)

	return r.mapping, nil
}

// failingResolver resolves nothing and always fails, emulating a resolver
// outage that endpoints and codecs must propagate.
type failingResolver struct{}

// ResolveModel implements [ModelResolver].
func (failingResolver) ResolveModel(
	context.Context,
	ResolveModelParams,
) (ModelMapping, error) {
	return ModelMapping{}, errors.New("resolver outage")
}

// sdkFailingResolver fails every lookup with a configuration SDKError,
// exercising the resolver's typed-error path.
type sdkFailingResolver struct{}

// ResolveModel implements [ModelResolver].
func (sdkFailingResolver) ResolveModel(
	context.Context,
	ResolveModelParams,
) (ModelMapping, error) {
	return ModelMapping{}, &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindConfiguration,
		Message: "lookup failure",
		Err:     nil,
	}
}

// TestMediaSniffingFallback pins the net/http byte-sniffing behavior the
// media-output codecs (CDN outputs that omit or garble Content-Type) and the
// speech-recognition codecs (file-part Content-Type and filename extension)
// rely on. If a Go release changes these signatures, the sniffing fallback in
// the queue exchange and openAIAudioExtension need revisiting.
func TestMediaSniffingFallback(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		wantMediaType string
		bytes         []byte
	}{
		{
			name:          "webm",
			wantMediaType: "video/webm",
			bytes:         []byte{0x1A, 0x45, 0xDF, 0xA3},
		},
		{
			name:          "png",
			wantMediaType: "image/png",
			bytes:         []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a},
		},
		{
			name:          "jpeg",
			wantMediaType: "image/jpeg",
			bytes:         []byte{0xFF, 0xD8, 0xFF, 0xE0},
		},
		{
			name:          "wav",
			wantMediaType: "audio/wave",
			bytes:         []byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"),
		},
		{
			name:          "mp3",
			wantMediaType: "audio/mpeg",
			bytes:         []byte("ID3\x03\x00\x00\x00\x00\x00\x00"),
		},
		{
			// MP4/M4A containers: net/http labels them video/mp4 when the
			// ftyp brands contain "mp4", regardless of audio content.
			name:          "mp4 container",
			wantMediaType: "video/mp4",
			bytes:         []byte("\x00\x00\x00\x18ftypisom\x00\x00\x02\x00mp42isom"),
		},
		{
			name:          "ogg",
			wantMediaType: "application/ogg",
			bytes:         []byte("OggS\x00\x02\x00\x00"),
		},
		{
			// net/http has no AMR signature either, and AMR files are
			// ASCII-ish: the sniffer reports text/plain, so AMR input falls
			// back to the ".wav" filename like any unrecognized format.
			name:          "amr (net/http undetected)",
			wantMediaType: "text/plain; charset=utf-8",
			bytes:         []byte("#!AMR\n"),
		},
		{
			// net/http has no FLAC signature: the sniffer reports
			// application/octet-stream, and the speech-recognition codec
			// checks the "fLaC" prefix explicitly to label FLAC input.
			name:          "flac (net/http undetected)",
			wantMediaType: "application/octet-stream",
			bytes:         []byte("fLaC\x00\x00\x00\x22"),
		},
		{
			name:          "unknown binary",
			wantMediaType: "application/octet-stream",
			bytes:         []byte{0x00, 0x01, 0x02, 0x03},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			require.Equal(t, tc.wantMediaType, http.DetectContentType(tc.bytes))
		})
	}
}

// assertImageMediaDecode exercises an image-output codec's Decode, shared by
// the queue media providers (WaveSpeed, fal.ai, Replicate): a valid image
// response is returned with its media type, and a non-image Content-Type is
// rejected as a serialization error. getPayload extracts the image bytes and
// advertised media type from the response.
func assertImageMediaDecode[Res any](
	t *testing.T,
	decode func(context.Context, DecodeParams) (Res, error),
	getPayload func(Res) ([]byte, string),
) {
	t.Helper()

	t.Run("valid image", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "image/png; charset=utf-8")
		body := []byte{0x89, 'P', 'N', 'G'}

		resp, err := decode(context.Background(), DecodeParams{
			Body:    body,
			Headers: headers,
		})
		require.NoError(t, err)

		image, mediaType := getPayload(resp)
		require.Equal(t, body, image)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("invalid content type", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Content-Type", "application/json")

		_, err := decode(context.Background(), DecodeParams{
			Body:    []byte(`{"foo":"bar"}`),
			Headers: headers,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
	})
}
