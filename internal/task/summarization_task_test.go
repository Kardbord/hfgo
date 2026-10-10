//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

func TestSummarize_ResponseVariations(t *testing.T) {
	t.Parallel()

	runSingleResponseVariations(
		t,
		[]singleResponseVariationCase{
			{
				name:         "single summary",
				responseBody: `[{"summary_text":"A concise summary."}]`,
				wantLen:      1,
				wantText:     "A concise summary.",
				description:  "a single summary is returned",
			},
			{
				name:         "empty response",
				responseBody: `[]`,
				wantLen:      0,
				description:  "empty response passes through as an empty list",
			},
			{
				name:         "multiple summaries",
				responseBody: `[{"summary_text":"One."},{"summary_text":"Two."}]`,
				wantLen:      2,
				wantText:     "One.",
				description:  "multiple summaries in one response pass through",
			},
		},
		func() hftypes.SummarizationRequest { return hftypes.SummarizationRequest{Input: "Some long text."} },
		func(c hfgo.Client, req hftypes.SummarizationRequest) ([]hftypes.Summarization, error) {
			return c.Summarize(req)
		},
		func(s hftypes.Summarization) string { return s.SummaryText },
	)
}

func TestSummarize_ParameterSerialization(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		cleanUp     *bool
		truncation  *string
		generate    map[string]any
		want        map[string]any
		description string
	}{
		{
			name:        "parameters omitted",
			description: "nil parameters are omitted from the request body",
		},
		{
			name:        "clean up tokenization spaces",
			cleanUp:     new(true),
			want:        map[string]any{"clean_up_tokenization_spaces": true},
			description: "clean_up_tokenization_spaces maps to its JSON key",
		},
		{
			name:        "truncation do_not_truncate",
			truncation:  new(hftypes.SummarizationTruncationDoNotTruncate),
			want:        map[string]any{"truncation": hftypes.SummarizationTruncationDoNotTruncate},
			description: "do_not_truncate constant serializes correctly",
		},
		{
			name:        "truncation longest_first",
			truncation:  new(hftypes.SummarizationTruncationLongestFirst),
			want:        map[string]any{"truncation": hftypes.SummarizationTruncationLongestFirst},
			description: "longest_first constant serializes correctly",
		},
		{
			name:        "truncation only_first",
			truncation:  new(hftypes.SummarizationTruncationOnlyFirst),
			want:        map[string]any{"truncation": hftypes.SummarizationTruncationOnlyFirst},
			description: "only_first constant serializes correctly",
		},
		{
			name:        "truncation only_second",
			truncation:  new(hftypes.SummarizationTruncationOnlySecond),
			want:        map[string]any{"truncation": hftypes.SummarizationTruncationOnlySecond},
			description: "only_second constant serializes correctly",
		},
		{
			name:     "generate parameters",
			generate: map[string]any{"max_new_tokens": 60, "temperature": 0.8},
			want: map[string]any{
				"generate_parameters": map[string]any{
					"max_new_tokens": float64(60),
					"temperature":    0.8,
				},
			},
			description: "generate_parameters maps to its JSON key",
		},
		{
			name:       "all parameters",
			cleanUp:    new(true),
			truncation: new(hftypes.SummarizationTruncationOnlyFirst),
			generate:   map[string]any{"max_new_tokens": 60},
			want: map[string]any{
				"clean_up_tokenization_spaces": true,
				"truncation":                   hftypes.SummarizationTruncationOnlyFirst,
				"generate_parameters":          map[string]any{"max_new_tokens": float64(60)},
			},
			description: "all parameters serialize together",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				`[{"summary_text":"A concise summary."}]`,
				nil,
			)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.SummarizationRequest{Input: "Some long text."}
			if tc.cleanUp != nil || tc.truncation != nil || tc.generate != nil {
				req.Parameters = &hftypes.SummarizationParameters{
					CleanUpTokenizationSpaces: tc.cleanUp,
					Truncation:                tc.truncation,
					GenerateParameters:        tc.generate,
				}
			}

			result, err := client.Summarize(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)

			reqBody := testutils.ReadRequestBody(t, mt)
			if tc.want == nil {
				_, ok := reqBody["parameters"]
				require.False(t, ok, tc.description)

				return
			}

			require.Equal(t, tc.want, reqBody["parameters"], tc.description)
		})
	}
}

func TestSummarize_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"summary_text":"A concise summary."}]`,
				want:         testutils.WantErrSDK,
				sdkErrKind:   hferrors.SDKErrorKindConfiguration,
				description:  "SDK error when model is missing",
			},
			{
				name:         "API error on 404",
				withModel:    true,
				statusCode:   http.StatusNotFound,
				responseBody: `{"error":"Model not found"}`,
				want:         testutils.WantErrAPI,
				description:  "API error for nonexistent model",
			},
		},
		func(opts ...hfopts.Option) ([]hftypes.Summarization, error) {
			return hfgo.NewClient(opts...).Summarize(hftypes.SummarizationRequest{
				Input: "Some long text that should be summarized.",
			})
		},
	)
}
