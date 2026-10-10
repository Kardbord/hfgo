//go:build integration

package integration_tests

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/stretchr/testify/require"
)

// TestTokenClassification_LiveAPI tests a basic token classification against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "dslim/bert-base-NER"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ClassifyTokens(
		hftypes.TokenClassificationRequest{
			Input: "My name is Sarah and I live in London.",
		},
	)

	require.NoError(t, err, "token classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have entities")

	for _, entity := range resp {
		require.NotEmpty(t, entity.Word, "entity should have a word")
		require.GreaterOrEqual(t, entity.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, entity.Score, 1.0, "score should be at most 1.0")
		require.GreaterOrEqual(t, entity.Start, 0, "start should be non-negative")
		require.Greater(t, entity.End, entity.Start, "end should be after start")

		t.Logf("Entity: %s (%s) at [%d:%d] score=%.4f",
			entity.Word,
			entityLabel(entity),
			entity.Start,
			entity.End,
			entity.Score,
		)
	}
}

// TestTokenClassification_WithAggregationStrategy tests token classification with various
// aggregation strategies against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_WithAggregationStrategy(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "dslim/bert-base-NER"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel(model),
		hfopts.WithContext(ctx),
	)

	strategies := []string{
		hftypes.TokenClassificationAggregationNone,
		hftypes.TokenClassificationAggregationSimple,
		hftypes.TokenClassificationAggregationFirst,
		hftypes.TokenClassificationAggregationAverage,
		hftypes.TokenClassificationAggregationMax,
	}

	for _, strategy := range strategies {
		t.Run(strategy, func(t *testing.T) {
			strat := strings.Clone(strategy)
			resp, err := client.ClassifyTokens(
				hftypes.TokenClassificationRequest{
					Input: "My name is Sarah and I live in London.",
					Parameters: &hftypes.TokenClassificationParameters{
						AggregationStrategy: &strat,
					},
				},
			)

			require.NoError(t, err, "token classification with strategy %q should succeed", strategy)
			require.NotNil(t, resp, "response should not be nil")
			require.NotEmpty(t, resp, "response should have entities")

			for _, entity := range resp {
				require.NotEmpty(t, entity.Word, "entity should have a word")
				require.GreaterOrEqual(t, entity.Score, 0.0, "score should be non-negative")
				require.LessOrEqual(t, entity.Score, 1.0, "score should be at most 1.0")

				if strategy == hftypes.TokenClassificationAggregationNone {
					require.NotNil(t, entity.Entity, "entity field should be set for strategy 'none'")
				} else {
					require.NotNil(t, entity.EntityGroup, "entity_group field should be set for strategy %q", strategy)
				}
			}

			t.Logf("Strategy %q: %d entities", strategy, len(resp))
		})
	}
}

// TestTokenClassification_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := hfgo.NewClient(
		hfopts.WithToken(apiToken),
		hfopts.WithModel("dslim/bert-base-NER"),
		hfopts.WithContext(ctx),
	)

	resp, err := client.ClassifyTokens(
		hftypes.TokenClassificationRequest{
			Input: "My name is Sarah and I live in London.",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// entityLabel returns the entity label from either Entity or EntityGroup field.
func entityLabel(e hftypes.TokenClassification) string {
	if e.EntityGroup != nil {
		return *e.EntityGroup
	}
	if e.Entity != nil {
		return *e.Entity
	}
	return ""
}
