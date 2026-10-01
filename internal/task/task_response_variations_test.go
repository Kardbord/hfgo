//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// singleResponseVariationCase describes one expected response shape for an
// endpoint whose API returns a flat list of text-bearing outputs.
type singleResponseVariationCase struct {
	// name is the subtest name.
	name string
	// responseBody is the mocked JSON response body.
	responseBody string
	// wantLen is the expected length of the returned list.
	wantLen int
	// wantText is the expected text of the first element, when wantLen > 0.
	wantText string
	// description explains the scenario; it is used in failure messages.
	description string
}

// runSingleResponseVariations drives a table of single-input response shapes
// against call, which performs one SDK request and returns its flat result
// list. text extracts the text field from a single element for comparison.
//
// The shared scaffolding (mock transport, client construction, len/text
// assertions) lives here so endpoints that share the flat-list-of-text output
// shape — e.g. translation and summarization — do not duplicate it.
func runSingleResponseVariations[Req, Res any](
	t *testing.T,
	cases []singleResponseVariationCase,
	buildReq func() Req,
	call func(client hfgo.Client, req Req) ([]Res, error),
	text func(Res) string,
) {
	t.Helper()

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			result, err := call(client, buildReq())
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)
			require.Len(t, result, tc.wantLen, tc.description)

			if tc.wantLen > 0 {
				require.Equal(t, tc.wantText, text(result[0]), tc.description)
			}
		})
	}
}

// batchResponseVariationCase describes one expected response shape for a
// batch call that returns a flat list of text-bearing outputs, one per input.
type batchResponseVariationCase struct {
	// name is the subtest name.
	name string
	// responseBody is the mocked JSON response body.
	responseBody string
	// want is the expected text of each element, in order.
	want []string
	// description explains the scenario; it is used in failure messages.
	description string
}

// runBatchResponseVariations drives a table of batch response shapes against
// call, which performs one SDK request and returns its flat result list. text
// extracts the text field from a single element for comparison.
//
// The shared scaffolding lives here for the same reason as
// runSingleResponseVariations: endpoints sharing the flat-list-of-text output
// shape should not duplicate it.
func runBatchResponseVariations[Req, Res any](
	t *testing.T,
	cases []batchResponseVariationCase,
	buildReq func() Req,
	call func(client hfgo.Client, req Req) ([]Res, error),
	text func(Res) string,
) {
	t.Helper()

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			result, err := call(client, buildReq())
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)
			require.Len(t, result, len(tc.want), tc.description)

			for j, r := range result {
				require.Equal(t, tc.want[j], text(r), tc.description)
			}
		})
	}
}
