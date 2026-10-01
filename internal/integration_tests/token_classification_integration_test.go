//go:build integration

package hfgo

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

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

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	resp, err := client.ClassifyTokens(
		TokenClassificationRequest{
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

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	strategies := []string{
		TokenClassificationAggregationNone,
		TokenClassificationAggregationSimple,
		TokenClassificationAggregationFirst,
		TokenClassificationAggregationAverage,
		TokenClassificationAggregationMax,
	}

	for _, strategy := range strategies {
		t.Run(strategy, func(t *testing.T) {
			strat := strings.Clone(strategy)
			resp, err := client.ClassifyTokens(
				TokenClassificationRequest{
					Input: "My name is Sarah and I live in London.",
					Parameters: &TokenClassificationParameters{
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

				if strategy == TokenClassificationAggregationNone {
					require.NotNil(t, entity.Entity, "entity field should be set for strategy 'none'")
				} else {
					require.NotNil(t, entity.EntityGroup, "entity_group field should be set for strategy %q", strategy)
				}
			}

			t.Logf("Strategy %q: %d entities", strategy, len(resp))
		})
	}
}

// TestTokenClassification_BatchLiveAPI tests batch token classification against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_BatchLiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "dslim/bert-base-NER"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	inputs := []string{
		"My name is Sarah and I live in London.",
		"I work at Google in New York.",
		"Angela Merkel was the Chancellor of Germany.",
	}

	resp, err := client.ClassifyTokensBatch(
		TokenClassificationBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch token classification should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, len(inputs), "response should have one entry per input")

	for i, entities := range resp {
		require.NotEmpty(t, entities, "each input should have entities")

		for _, entity := range entities {
			require.NotEmpty(t, entity.Word, "entity should have a word")
			require.GreaterOrEqual(t, entity.Score, 0.0, "score should be non-negative")
			require.LessOrEqual(t, entity.Score, 1.0, "score should be at most 1.0")
		}

		t.Logf("Input %d: %d entities", i, len(entities))
	}
}

// TestTokenClassification_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("dslim/bert-base-NER"),
		WithContext(ctx),
	)

	resp, err := client.ClassifyTokens(
		TokenClassificationRequest{
			Input: "My name is Sarah and I live in London.",
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}

// TestTokenClassification_VeryLargeBatch tests token classification with a larger batch of inputs.
// This test requires the HF_TOKEN environment variable to be set.
func TestTokenClassification_VeryLargeBatch(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("dslim/bert-base-NER"),
		WithContext(ctx),
	)

	inputs := []string{
		"My name is Sarah and I live in London.",
		"I work at Google in New York.",
		"Angela Merkel was the Chancellor of Germany.",
		"Apple Inc. is headquartered in Cupertino, California.",
		"Barack Obama was the 44th President of the United States.",
		"The Eiffel Tower is located in Paris, France.",
		"Amazon was founded by Jeff Bezos in Seattle.",
		"Tokyo is the capital city of Japan.",
		"Marie Curie won the Nobel Prize in Physics.",
		"The United Nations is based in New York City.",
	}

	resp, err := client.ClassifyTokensBatch(
		TokenClassificationBatchRequest{
			Inputs: inputs,
		},
	)

	require.NoError(t, err, "batch token classification with larger batch should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.Len(t, resp, len(inputs), "response should have one entry per input")

	for _, entities := range resp {
		require.NotEmpty(t, entities, "each input should have entities")
	}
}

// entityLabel returns the entity label from either Entity or EntityGroup field.
func entityLabel(e TokenClassification) string {
	if e.EntityGroup != nil {
		return *e.EntityGroup
	}
	if e.Entity != nil {
		return *e.Entity
	}
	return ""
}
