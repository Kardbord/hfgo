//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// mediaDecodingCase describes one binary-media response decoding scenario: a
// mocked response body with its advertised Content-Type (empty means unset),
// and the expected outcome. Shared by the text-to-video and image-to-image
// decoding tables.
type mediaDecodingCase struct {
	name          string
	response      []byte
	contentType   string
	wantMediaType string
	wantErr       bool
	description   string
}

// newMediaTransport builds a mock transport returning body with the given
// Content-Type. An empty contentType leaves the header unset.
func newMediaTransport(body []byte, contentType string) *testutils.MockTransport {
	mt := testutils.NewMockTransport(http.StatusOK, string(body), nil)
	if contentType != "" {
		mt.Response.Header.Set("Content-Type", contentType)
	}

	return mt
}

// runMediaDecodingCases runs each mediaDecodingCase against run, which must
// issue the task request against a mock transport built from the case and
// return the task response plus error. getBytes and getMediaType extract the
// raw payload and advertised media type from the response for the success
// assertions; failures must assert a serialization SDK error and a zero
// response.
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
				testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)

				var zero Res
				require.Equal(t, zero, resp, tc.description)

				return
			}

			require.NoError(t, err, tc.description)
			require.Equal(t, tc.response, getBytes(resp), tc.description)
			require.Equal(t, tc.wantMediaType, getMediaType(resp), tc.description)
		})
	}
}
