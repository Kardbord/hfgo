//go:build !integration

package task_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/stretchr/testify/require"
)

// mediaDecodingCase describes one binary-media response decoding scenario: the
// mocked output bytes with their advertised Content-Type (empty means unset,
// exercising the sniffing fallback), and the expected outcome. Shared by the
// text-to-video and image-to-image decoding tables.
type mediaDecodingCase struct {
	name          string
	output        []byte
	outputType    string
	wantMediaType string
	wantErr       bool
	description   string
}

// newMediaServer builds a test server exercising the async exchange's media
// handling: the submit response completes immediately (WaveSpeed's sync-mode
// path) and points at an output endpoint serving the case's bytes and content
// type. An empty outputType leaves the Content-Type unset.
func newMediaServer(tc mediaDecodingCase) *httptest.Server {
	var outputURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			// Submit: the task is already complete, so the exchange goes
			// straight to the output download.
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(
				w,
				`{"code":200,"message":"success","data":{"id":"pred_1","status":"completed","outputs":["%s"]}}`,
				outputURL,
			)

			return
		}

		if tc.outputType != "" {
			w.Header().Set("Content-Type", tc.outputType)
		}
		_, _ = w.Write(tc.output)
	}))
	outputURL = server.URL + "/output"

	return server
}

// runMediaDecodingCases runs each mediaDecodingCase against run, which must
// issue the task request against a server built from the case and return the
// task response plus error. getBytes and getMediaType extract the raw payload
// and advertised media type from the response for the success assertions;
// failures must assert a serialization SDK error and a zero response.
func runMediaDecodingCases[Res any](
	t *testing.T,
	cases []mediaDecodingCase,
	run func(tc mediaDecodingCase) (Res, error),
	getBytes func(Res) []byte,
	getMediaType func(Res) string,
) {
	t.Helper()

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp, err := run(tc)

			if tc.wantErr {
				require.Error(t, err, tc.description)
				requireAsyncTaskErrorKind(t, err)

				var zero Res
				require.Equal(t, zero, resp, tc.description)

				return
			}

			require.NoError(t, err, tc.description)
			require.Equal(t, tc.output, getBytes(resp), tc.description)
			require.Equal(t, tc.wantMediaType, getMediaType(resp), tc.description)
		})
	}
}

// requireAsyncTaskErrorKind asserts err is a serialization SDK error, the kind
// the WaveSpeed exchange surfaces for undecodable media.
func requireAsyncTaskErrorKind(t *testing.T, err error) {
	t.Helper()

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindSerialization, sdkErr.Kind)
}
