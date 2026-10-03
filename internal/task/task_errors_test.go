package task_test

import (
	"net/http"
	"reflect"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
)

// errorCase describes a single expected-error scenario for a task function call.
// It is shared by the per-task error-table tests.
type errorCase struct {
	// name is the subtest name.
	name string
	// withModel reports whether a model option should be set for the call.
	withModel bool
	// statusCode is the mocked HTTP status code returned by the transport.
	statusCode int
	// responseBody is the mocked response body returned by the transport.
	responseBody string
	// want selects whether an SDK or API error is expected.
	want testutils.WantErr
	// sdkErrKind is the expected error kind when want is WantErrSDK.
	sdkErrKind hferrors.SDKErrorKind
	// description explains the scenario; it is used in failure messages.
	description string
}

// runErrorCases runs each errorCase against run, asserting the expected error
// type. For each case it builds options backed by a mock transport (with a
// model set only when withModel is true), invokes run, and verifies that:
//
//   - an error is returned,
//   - the result is nil,
//   - it is of the expected type (SDKError kind or APIError status), and
//   - SDK configuration errors make no request.
//
// run issues a single task function call with the given options and returns its
// result. Keeping the table loop here avoids duplicating it across task functions.
func runErrorCases[Res any](
	t *testing.T,
	cases []errorCase,
	run func(opts ...hfopts.Option) (Res, error),
) {
	t.Helper()

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(tc.statusCode, tc.responseBody, nil)
			opts := []hfopts.Option{
				hfopts.WithHTTPClientFactory(func() http.Client {
					return testutils.NewMockHTTPClient(mt)
				}),
			}
			if tc.withModel {
				opts = append(opts, hfopts.WithModel("nonexistent-model"))
			}

			result, err := run(opts...)
			if err == nil {
				t.Fatalf("expected an error: %s", tc.description)
			}
			if !reflect.ValueOf(result).IsZero() {
				t.Errorf("expected zero result, got %v: %s", result, tc.description)
			}

			if tc.want == testutils.WantErrSDK {
				testutils.AssertSDKErrorKind(t, err, tc.sdkErrKind)
				if mt.LastRequest != nil {
					t.Fatal("SDK config errors short-circuit before any request")
				}
			} else {
				testutils.AssertAPIErrorStatus(t, err, tc.statusCode)
			}
		})
	}
}
