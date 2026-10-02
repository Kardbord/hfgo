//go:build !integration

package task_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/internal/dto"
	"github.com/Kardbord/hfgo/v4/internal/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
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
		func() dto.SummarizationRequest { return dto.SummarizationRequest{Input: "Some long text."} },
		func(c hfgo.Client, req dto.SummarizationRequest) ([]dto.Summarization, error) {
			return c.Summarize(req)
		},
		func(s dto.Summarization) string { return s.SummaryText },
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
			cleanUp:     testutils.Ptr(true),
			want:        map[string]any{"clean_up_tokenization_spaces": true},
			description: "clean_up_tokenization_spaces maps to its JSON key",
		},
		{
			name:        "truncation do_not_truncate",
			truncation:  testutils.Ptr(dto.SummarizationTruncationDoNotTruncate),
			want:        map[string]any{"truncation": dto.SummarizationTruncationDoNotTruncate},
			description: "do_not_truncate constant serializes correctly",
		},
		{
			name:        "truncation longest_first",
			truncation:  testutils.Ptr(dto.SummarizationTruncationLongestFirst),
			want:        map[string]any{"truncation": dto.SummarizationTruncationLongestFirst},
			description: "longest_first constant serializes correctly",
		},
		{
			name:        "truncation only_first",
			truncation:  testutils.Ptr(dto.SummarizationTruncationOnlyFirst),
			want:        map[string]any{"truncation": dto.SummarizationTruncationOnlyFirst},
			description: "only_first constant serializes correctly",
		},
		{
			name:        "truncation only_second",
			truncation:  testutils.Ptr(dto.SummarizationTruncationOnlySecond),
			want:        map[string]any{"truncation": dto.SummarizationTruncationOnlySecond},
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
			cleanUp:    testutils.Ptr(true),
			truncation: testutils.Ptr(dto.SummarizationTruncationOnlyFirst),
			generate:   map[string]any{"max_new_tokens": 60},
			want: map[string]any{
				"clean_up_tokenization_spaces": true,
				"truncation":                   dto.SummarizationTruncationOnlyFirst,
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
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			req := dto.SummarizationRequest{Input: "Some long text."}
			if tc.cleanUp != nil || tc.truncation != nil || tc.generate != nil {
				req.Parameters = &dto.SummarizationParameters{
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
		func(opts ...request.Option) ([]dto.Summarization, error) {
			return hfgo.NewClient(opts...).Summarize(dto.SummarizationRequest{
				Input: "Some long text that should be summarized.",
			})
		},
	)
}

func TestSummarizeBatch_ResponseVariations(t *testing.T) {
	t.Parallel()

	runBatchResponseVariations(
		t,
		[]batchResponseVariationCase{
			{
				name:         "single input",
				responseBody: `[{"summary_text":"Summary one."}]`,
				want:         []string{"Summary one."},
				description:  "a single batched input returns one summary",
			},
			{
				name:         "multiple inputs",
				responseBody: `[{"summary_text":"Summary one."},{"summary_text":"Summary two."}]`,
				want:         []string{"Summary one.", "Summary two."},
				description:  "each batched input returns its own flat summary",
			},
			{
				name:         "empty response",
				responseBody: `[]`,
				want:         []string{},
				description:  "empty response passes through as an empty list",
			},
		},
		func() dto.SummarizationBatchRequest {
			return dto.SummarizationBatchRequest{
				Inputs: []string{"Long text one.", "Long text two."},
			}
		},
		func(c hfgo.Client, req dto.SummarizationBatchRequest) ([]dto.Summarization, error) {
			return c.SummarizeBatch(req)
		},
		func(s dto.Summarization) string { return s.SummaryText },
	)
}

func TestSummarizeBatch_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"summary_text":"Summary one."}]`,
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
		func(opts ...request.Option) ([]dto.Summarization, error) {
			return hfgo.NewClient(opts...).SummarizeBatch(dto.SummarizationBatchRequest{
				Inputs: []string{"Long text one."},
			})
		},
	)
}

func TestSummarizeBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"summary_text":"Summary one."}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	result, err := client.SummarizeBatch(dto.SummarizationBatchRequest{
		Inputs: []string{"Long text one."},
	}, hfgo.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify the correct model was used in the request
	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")
}
