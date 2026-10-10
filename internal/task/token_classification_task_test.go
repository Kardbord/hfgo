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

func TestClassifyTokens_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		responseBody string
		expectedLen  int
		wantEntity   *string
		wantGroup    *string
		wantScore    float64
		wantWord     string
		wantStart    int
		wantEnd      int
		description  string
	}{
		{
			name:         "single entity with entity field",
			responseBody: `[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
			expectedLen:  1,
			wantEntity:   new("PER"),
			wantGroup:    nil,
			wantScore:    0.998,
			wantWord:     "Sarah",
			wantStart:    11,
			wantEnd:      16,
			description:  "single entity with entity field (aggregation_strategy=none)",
		},
		{
			name:         "single entity with entity_group field",
			responseBody: `[{"entity_group":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
			expectedLen:  1,
			wantEntity:   nil,
			wantGroup:    new("PER"),
			wantScore:    0.998,
			wantWord:     "Sarah",
			wantStart:    11,
			wantEnd:      16,
			description:  "single entity with entity_group field (aggregation_strategy=simple)",
		},
		{
			name:         "multiple entities",
			responseBody: `[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16},{"entity":"LOC","score":0.995,"word":"London","start":31,"end":37}]`,
			expectedLen:  2,
			wantEntity:   new("PER"),
			wantGroup:    nil,
			wantScore:    0.998,
			wantWord:     "Sarah",
			wantStart:    11,
			wantEnd:      16,
			description:  "multiple entities preserve order",
		},
		{
			name:         "empty response",
			responseBody: `[]`,
			expectedLen:  0,
			description:  "input with no entities",
		},
	}

	for i := range cases {
		tc := cases[i]
		t.Run(tc.name, func(t *testing.T) {
			mt := testutils.NewJSONMockTransport(http.StatusOK, tc.responseBody, nil)
			client := hfgo.NewClient(
				hfopts.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfopts.WithModel("test-model"),
			)

			req := hftypes.TokenClassificationRequest{
				Input: "My name is Sarah and I live in London.",
			}

			result, err := client.ClassifyTokens(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedLen, tc.description)

			if tc.expectedLen > 0 {
				entity := result[0]
				if tc.wantEntity != nil {
					require.NotNil(t, entity.Entity)
					require.Equal(t, *tc.wantEntity, *entity.Entity)
				} else {
					require.Nil(t, entity.Entity)
				}
				if tc.wantGroup != nil {
					require.NotNil(t, entity.EntityGroup)
					require.Equal(t, *tc.wantGroup, *entity.EntityGroup)
				} else {
					require.Nil(t, entity.EntityGroup)
				}
				require.InEpsilon(t, tc.wantScore, entity.Score, 0.001)
				require.Equal(t, tc.wantWord, entity.Word)
				require.Equal(t, tc.wantStart, entity.Start)
				require.Equal(t, tc.wantEnd, entity.End)
			}
		})
	}
}

func TestClassifyTokens_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"entity_group":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfopts.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfopts.WithModel("test-model"),
	)

	req := hftypes.TokenClassificationRequest{
		Input: "My name is Sarah and I live in London.",
		Parameters: &hftypes.TokenClassificationParameters{
			IgnoreLabels:        []string{"O"},
			Stride:              new(5),
			AggregationStrategy: new(hftypes.TokenClassificationAggregationSimple),
		},
	}

	result, err := client.ClassifyTokens(req)
	require.NoError(t, err)
	require.NotNil(t, result)

	reqBody := testutils.ReadRequestBody(t, mt)
	params, ok := reqBody["parameters"].(map[string]any)
	require.True(t, ok, "parameters should be a map")
	require.Equal(t, "simple", params["aggregation_strategy"])
	require.InEpsilon(t, float64(5), params["stride"], 0.001)

	ignoreLabels, ok := params["ignore_labels"].([]any)
	require.True(t, ok, "ignore_labels should be a list")
	require.Len(t, ignoreLabels, 1)
	require.Equal(t, "O", ignoreLabels[0])
}

func TestClassifyTokens_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
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
		func(opts ...hfopts.Option) ([]hftypes.TokenClassification, error) {
			return hfgo.NewClient(opts...).ClassifyTokens(hftypes.TokenClassificationRequest{
				Input: "My name is Sarah and I live in London.",
			})
		},
	)
}
