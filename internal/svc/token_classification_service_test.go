//go:build !integration

package svc_test

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

func TestTokenClassificationService_ClassifyTokens_ResponseDecoding(t *testing.T) {
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
			wantEntity:   testutils.Ptr("PER"),
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
			wantGroup:    testutils.Ptr("PER"),
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
			wantEntity:   testutils.Ptr("PER"),
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
				hfgo.WithHTTPClientFactory(
					func() http.Client { return testutils.NewMockHTTPClient(mt) },
				),
				hfgo.WithModel("test-model"),
			)

			req := dto.TokenClassificationRequest{
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

func TestTokenClassificationService_ClassifyTokens_WithParameters(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[{"entity_group":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
		hfgo.WithModel("test-model"),
	)

	req := dto.TokenClassificationRequest{
		Input: "My name is Sarah and I live in London.",
		Parameters: &dto.TokenClassificationParameters{
			IgnoreLabels:        []string{"O"},
			Stride:              testutils.Ptr(5),
			AggregationStrategy: testutils.Ptr(dto.TokenClassificationAggregationSimple),
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

func TestTokenClassificationService_ClassifyTokens_Errors(t *testing.T) {
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
		func(opts ...request.Option) ([]dto.TokenClassification, error) {
			return hfgo.NewClient(opts...).ClassifyTokens(dto.TokenClassificationRequest{
				Input: "My name is Sarah and I live in London.",
			})
		},
	)
}

func TestTokenClassificationService_ClassifyTokensBatch_ResponseDecoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name             string
		responseBody     string
		inputs           []string
		expectedOuterLen int
		expectedInnerLen []int
		wantEntity       *string
		wantGroup        *string
		wantScore        float64
		wantWord         string
		wantStart        int
		wantEnd          int
		description      string
	}{
		{
			name:             "single input in batch",
			responseBody:     `[[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]]`,
			inputs:           []string{"My name is Sarah."},
			expectedOuterLen: 1,
			expectedInnerLen: []int{1},
			wantEntity:       testutils.Ptr("PER"),
			wantScore:        0.998,
			wantWord:         "Sarah",
			wantStart:        11,
			wantEnd:          16,
			description:      "single batched token classification",
		},
		{
			name:         "multiple inputs",
			responseBody: `[[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}],[{"entity":"ORG","score":0.995,"word":"Google","start":10,"end":16}]]`,
			inputs: []string{
				"My name is Sarah.",
				"I work at Google.",
			},
			expectedOuterLen: 2,
			expectedInnerLen: []int{1, 1},
			wantEntity:       testutils.Ptr("PER"),
			wantScore:        0.998,
			wantWord:         "Sarah",
			wantStart:        11,
			wantEnd:          16,
			description:      "multiple batched token classifications",
		},
		{
			name:         "multiple entities per input",
			responseBody: `[[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16},{"entity":"LOC","score":0.995,"word":"London","start":31,"end":37}],[{"entity":"ORG","score":0.995,"word":"Google","start":10,"end":16}]]`,
			inputs: []string{
				"My name is Sarah and I live in London.",
				"I work at Google.",
			},
			expectedOuterLen: 2,
			expectedInnerLen: []int{2, 1},
			wantEntity:       testutils.Ptr("PER"),
			wantScore:        0.998,
			wantWord:         "Sarah",
			wantStart:        11,
			wantEnd:          16,
			description:      "each batched input returns a list of entities",
		},
		{
			name:         "multiple inputs with entity_group",
			responseBody: `[[{"entity_group":"PER","score":0.998,"word":"Sarah","start":11,"end":16}],[{"entity_group":"ORG","score":0.995,"word":"Google","start":10,"end":16}]]`,
			inputs: []string{
				"My name is Sarah.",
				"I work at Google.",
			},
			expectedOuterLen: 2,
			expectedInnerLen: []int{1, 1},
			wantGroup:        testutils.Ptr("PER"),
			wantScore:        0.998,
			wantWord:         "Sarah",
			wantStart:        11,
			wantEnd:          16,
			description:      "batched token classification with entity_group (aggregation)",
		},
		{
			name:             "empty inner result",
			responseBody:     `[[]]`,
			inputs:           []string{"No entities here."},
			expectedOuterLen: 1,
			expectedInnerLen: []int{0},
			description:      "input with zero entities",
		},
	}

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

			req := dto.TokenClassificationBatchRequest{
				Inputs: tc.inputs,
			}

			result, err := client.ClassifyTokensBatch(req)
			require.NoError(t, err, tc.description)
			require.NotNil(t, result)
			require.Len(t, result, tc.expectedOuterLen, tc.description)

			for j := range result {
				require.Len(t, result[j], tc.expectedInnerLen[j], tc.description)
			}
			if len(result) > 0 && len(result[0]) > 0 {
				entity := result[0][0]
				if tc.wantEntity != nil {
					require.NotNil(t, entity.Entity)
					require.Equal(t, *tc.wantEntity, *entity.Entity)
				}
				if tc.wantGroup != nil {
					require.NotNil(t, entity.EntityGroup)
					require.Equal(t, *tc.wantGroup, *entity.EntityGroup)
				}
				require.InEpsilon(t, tc.wantScore, entity.Score, 0.001)
				require.Equal(t, tc.wantWord, entity.Word)
				require.Equal(t, tc.wantStart, entity.Start)
				require.Equal(t, tc.wantEnd, entity.End)
			}
		})
	}
}

func TestTokenClassificationService_ClassifyTokensBatch_Errors(t *testing.T) {
	t.Parallel()

	runErrorCases(t,
		[]errorCase{
			{
				name:         "no model configured",
				statusCode:   http.StatusOK,
				responseBody: `[[]]`,
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
		func(opts ...request.Option) ([][]dto.TokenClassification, error) {
			return hfgo.NewClient(opts...).ClassifyTokensBatch(dto.TokenClassificationBatchRequest{
				Inputs: []string{"My name is Sarah and I live in London."},
			})
		},
	)
}

//nolint:dupl // batch model-from-options test is structurally similar across domains
func TestTokenClassificationService_ClassifyTokensBatch_ModelFromOptions(t *testing.T) {
	t.Parallel()

	mt := testutils.NewJSONMockTransport(
		http.StatusOK,
		`[[{"entity":"PER","score":0.998,"word":"Sarah","start":11,"end":16}]]`,
		nil,
	)
	client := hfgo.NewClient(
		hfgo.WithHTTPClientFactory(func() http.Client { return testutils.NewMockHTTPClient(mt) }),
	)

	req := dto.TokenClassificationBatchRequest{
		Inputs: []string{"My name is Sarah."},
	}

	result, err := client.ClassifyTokensBatch(req, hfgo.WithModel("override-model"))
	require.NoError(t, err)
	require.NotNil(t, result)

	require.NotNil(t, mt.LastRequest)
	require.Contains(t, mt.LastRequest.URL.Path, "override-model")

	reqBody := testutils.ReadRequestBody(t, mt)
	inputs, ok := reqBody["inputs"].([]any)
	require.True(t, ok, "inputs should be an array for batched requests")
	require.Len(t, inputs, 1)
	require.Equal(t, "My name is Sarah.", inputs[0])
}
