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

func TestTranslate_ResponseVariations(t *testing.T) {
	t.Parallel()

	runSingleResponseVariations(
		t,
		[]singleResponseVariationCase{
			{
				name:         "single translation",
				responseBody: `[{"translation_text":"Bonjour le monde."}]`,
				wantLen:      1,
				wantText:     "Bonjour le monde.",
				description:  "a single translation is returned in a one-element list",
			},
			{
				name:         "multiple translations",
				responseBody: `[{"translation_text":"Bonjour."},{"translation_text":"Salut."}]`,
				wantLen:      2,
				wantText:     "Bonjour.",
				description:  "multiple candidates preserve order",
			},
			{
				name:         "empty response",
				responseBody: `[]`,
				wantLen:      0,
				description:  "empty response passes through as an empty list",
			},
		},
		func() hftypes.TranslationRequest { return hftypes.TranslationRequest{Input: "Hello world."} },
		func(c hfgo.Client, req hftypes.TranslationRequest) ([]hftypes.Translation, error) {
			return c.Translate(req)
		},
		func(tr hftypes.Translation) string { return tr.TranslationText },
	)
}

func TestTranslate_ParameterSerialization(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name        string
		params      *hftypes.TranslationParameters
		want        map[string]any
		description string
	}{
		{
			name:        "parameters omitted",
			description: "nil parameters are omitted from the request body",
		},
		{
			name: "source and target language",
			params: &hftypes.TranslationParameters{
				SrcLang: new("en"),
				TgtLang: new("fr"),
			},
			want:        map[string]any{"src_lang": "en", "tgt_lang": "fr"},
			description: "src_lang and tgt_lang map to their JSON keys",
		},
		{
			name: "clean up tokenization spaces",
			params: &hftypes.TranslationParameters{
				CleanUpTokenizationSpaces: new(true),
			},
			want:        map[string]any{"clean_up_tokenization_spaces": true},
			description: "clean_up_tokenization_spaces maps to its JSON key",
		},
		{
			name: "truncation",
			params: &hftypes.TranslationParameters{
				Truncation: new(hftypes.TranslationTruncationOnlyFirst),
			},
			want:        map[string]any{"truncation": hftypes.TranslationTruncationOnlyFirst},
			description: "truncation constant serializes correctly",
		},
		{
			name: "generate parameters",
			params: &hftypes.TranslationParameters{
				GenerateParameters: map[string]any{"max_new_tokens": 60},
			},
			want: map[string]any{
				"generate_parameters": map[string]any{"max_new_tokens": float64(60)},
			},
			description: "generate_parameters maps to its JSON key",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(
				http.StatusOK,
				`[{"translation_text":"Bonjour le monde."}]`,
				nil,
			)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.TranslationRequest{Input: "Hello world."}
			if tc.params != nil {
				req.Parameters = tc.params
			}

			result, err := client.Translate(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result, tc.description)

			reqBody := testutils.ReadRequestBody(t, mt)
			require.Equal(t, "Hello world.", reqBody["inputs"], tc.description)

			if tc.want == nil {
				_, ok := reqBody["parameters"]
				require.False(t, ok, tc.description)

				return
			}

			require.Equal(t, tc.want, reqBody["parameters"], tc.description)
		})
	}
}

func TestTranslate_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"translation_text":"Bonjour."}]`,
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
			{
				name:         "API error on 503",
				withModel:    true,
				statusCode:   http.StatusServiceUnavailable,
				responseBody: `{"error":"Model loading"}`,
				want:         testutils.WantErrAPI,
				description:  "API error for model not yet loaded",
			},
		},
		func(opts ...hfopts.Option) ([]hftypes.Translation, error) {
			return hfgo.NewClient(opts...).Translate(hftypes.TranslationRequest{
				Input: "Hello world.",
			})
		},
	)
}
