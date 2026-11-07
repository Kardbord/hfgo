//go:build !integration

package hfproviders

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// FuzzProviderCodecs drives arbitrary response bodies through a table of
// representative provider codecs and checks decode invariants: no panic on
// any byte sequence, errors are always well-typed *hferrors.SDKError values
// (never bare or foreign errors), and a successful decode returns a response
// that survives re-marshaling (no cyclic or otherwise invalid decoded
// values). The table covers the generic JSONCodec path, the hand-rolled
// fallback codecs (flat/nested shapes, output-extraction chains), and a
// media-prefix codec exercising the byte-sniffing path.
func FuzzProviderCodecs(f *testing.F) {
	decodeCalls := []struct {
		name     string
		decode   func(context.Context, DecodeParams) (any, error)
		ctHeader string // Content-Type the pipeline would present
	}{
		{
			name: "JSONCodec_ChatResponse",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return JSONCodec[hftypes.ChatRequest, hftypes.ChatResponse]{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "hfQuestionAnsweringCodec",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return hfQuestionAnsweringCodec{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "hfTextClassificationCodec",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return hfTextClassificationCodec{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "replicateSpeechRecognitionCodec",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return replicateSpeechRecognitionCodec{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "falaiSpeechRecognitionCodec",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return falaiSpeechRecognitionCodec{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "deepinfraSpeechRecognitionCodec",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				return deepinfraSpeechRecognitionCodec{}.Decode(ctx, params)
			},
			ctHeader: mimeTypeJSON,
		},
		{
			name: "falaiTextToImageCodec_media",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				codec := falaiTextToImageCodec{falaiQueueCodec{queueTaskCodec{
					mediaPrefix: falaiImageMediaPrefix,
				}}}

				return codec.Decode(ctx, params)
			},
			ctHeader: "image/png",
		},
		{
			name: "falaiTextToImageCodec_sniff",
			decode: func(ctx context.Context, params DecodeParams) (any, error) {
				codec := falaiTextToImageCodec{falaiQueueCodec{queueTaskCodec{
					mediaPrefix: falaiImageMediaPrefix,
				}}}

				return codec.Decode(ctx, params)
			},
			// Wrong advertised type: forces the http.DetectContentType
			// byte-sniffing fallback in queueDecodeMediaBody.
			ctHeader: "application/octet-stream",
		},
	}

	seeds := [][]byte{
		[]byte(`{"answer":"42","score":0.99}`),
		[]byte(`[{"answer":"42","score":0.99}]`),
		[]byte(`[{"label":"positive","score":0.95}]`),
		[]byte(`[[{"label":"positive","score":0.95}]]`),
		[]byte(`{"text":"hello world"}`),
		[]byte(`{"text":"hello","chunks":[{"text":"a","timestamp":[0.0,1.5]}]}`),
		[]byte(`{"text":null}`),
		[]byte(`{"segments":null,"text":"x"}`),
		[]byte(`{"segments":[{"start":1,"end":2,"text":"a"}],"text":"x"}`),
		[]byte(`{"status":"COMPLETED"}`),
		[]byte(
			`{"id":"1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
		),
		{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 1, 2, 3}, // PNG magic + junk
		[]byte(
			"\xff\xd8\xff\xe0 junk",
		), // JPEG magic (sniff must reject for image/ prefix... it matches image/)
		[]byte("<html>not an image</html>"),
		[]byte(`null`),
		[]byte(`[]`),
		[]byte(`{}`),
		[]byte(`"a string"`),
		[]byte(`42`),
		[]byte(`true`),
		[]byte(`{"a":{"b":{"c":{"d":{"e":[1,2,3]}}}}}`),
		[]byte(`{"answer":null}`),
		[]byte(`{"answer":42}`),
		[]byte(`{"transcription":"hi","translation":"hola"}`),
		[]byte(`{"txt_file":"http://example.invalid/x.txt"}`),
		[]byte(`[null]`),
		[]byte(`{"output":"abc"}`),
		[]byte(""),
		[]byte(`{"text":123}`),
		[]byte(`{"segments":"notanarray","text":"x"}`),
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, body []byte) {
		for _, call := range decodeCalls {
			params := DecodeParams{
				Body: body,
				Headers: http.Header{
					"Content-Type": []string{call.ctHeader},
				},
			}

			resp, err := call.decode(context.Background(), params)
			if err != nil {
				if _, ok := errors.AsType[*hferrors.SDKError](err); !ok {
					t.Errorf("%s: error is not a *hferrors.SDKError: %v (%T)", call.name, err, err)
				}

				continue
			}

			if _, err := json.Marshal(resp); err != nil {
				t.Errorf("%s: successfully decoded response cannot re-marshal: %v", call.name, err)
			}
		}
	})
}

// FuzzContentTypeClassification checks the Content-Type guards the codec
// layer relies on: parseMediaType and isJSONContentType never panic on
// arbitrary header values, malformed values never classify as JSON, and
// empty values never classify as JSON.
func FuzzContentTypeClassification(f *testing.F) {
	seeds := []string{
		"",
		"application/json",
		"application/json; charset=utf-8",
		"application/vnd.api+json",
		"text/event-stream",
		"image/png",
		"APPLICATION/JSON",
		"garbage",
		"/",
		";",
		"application/json;;;",
		"application/json; charset=",
		"application/json; =x",
		"text/plain;boundary=\x00",
		"application/json; q=\"unterminated",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, contentType string) {
		mediaType, ok := parseMediaType(contentType)

		if contentType == "" && ok {
			t.Errorf("parseMediaType(%q) reported ok for empty input", contentType)
		}

		classified := isJSONContentType(contentType)
		if classified && !ok {
			t.Errorf("isJSONContentType(%q) true but parseMediaType failed", contentType)
		}
		if classified && mediaType != mimeTypeJSON && !jsonSuffix(mediaType) {
			t.Errorf(
				"isJSONContentType(%q) true for non-JSON media type %q",
				contentType,
				mediaType,
			)
		}
	})
}

// jsonSuffix reports whether mediaType is a structured +json suffix type,
// mirroring isJSONContentType's acceptance rule.
func jsonSuffix(mediaType string) bool {
	return strings.HasSuffix(mediaType, "+json") && len(mediaType) > len("+json")
}
