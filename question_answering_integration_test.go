//go:build integration

package hfgo

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestQuestionAnswering_LiveAPI tests a basic question answering request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestQuestionAnswering_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "deepset/roberta-base-squad2"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	resp, err := client.AnswerQuestion(
		QuestionAnsweringRequest{
			Input: QuestionAnsweringInput{
				Question: "What is the capital of France?",
				Context:  "France is a country in Europe. Its capital is Paris. It is known for the Eiffel Tower.",
			},
		},
	)

	require.NoError(t, err, "question answering should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have answers")

	answer := resp[0]
	require.NotEmpty(t, answer.Answer, "answer should not be empty")
	require.GreaterOrEqual(t, answer.Score, 0.0, "score should be non-negative")
	require.LessOrEqual(t, answer.Score, 1.0, "score should be at most 1.0")
	require.GreaterOrEqual(t, answer.Start, 0, "start should be non-negative")
	require.Greater(t, answer.End, answer.Start, "end should be after start")

	t.Logf("Answer: %q (score: %.4f, span: [%d:%d])", answer.Answer, answer.Score, answer.Start, answer.End)
}

// TestQuestionAnswering_TopKResponseFormatQuirk documents the HuggingFace API quirk where
// the response format differs based on whether the top_k parameter is set.
//
// This test documents an important API quirk:
//   - When top_k is unset or 1: The API returns a bare JSON object {"answer":"...","score":...,"start":...,"end":...}
//   - When top_k is > 1: The API returns a JSON array [{"answer":"...","score":...},...]
//
// The SDK handles this transparently in the answer() method by dispatching
// to doJSONInference with the appropriate response type based on TopK.
//
// This test requires the HF_TOKEN environment variable to be set.
func TestQuestionAnswering_TopKResponseFormatQuirk(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "deepset/roberta-base-squad2"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	question := "What is the capital of France?"
	context := "France is a country in Europe. Its capital is Paris. It is known for the Eiffel Tower."

	// Test 1: Without top_k (unset) - API returns bare object, SDK normalizes to slice
	t.Run("without_topk_bare_object", func(t *testing.T) {
		resp, err := client.AnswerQuestion(
			QuestionAnsweringRequest{
				Input: QuestionAnsweringInput{
					Question: question,
					Context:  context,
				},
			},
		)

		require.NoError(t, err, "question answering without top_k should succeed")
		require.NotNil(t, resp, "response should not be nil")
		require.NotEmpty(t, resp, "response should have answers")

		answer := resp[0]
		require.NotEmpty(t, answer.Answer, "answer should not be empty")
		require.GreaterOrEqual(t, answer.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, answer.Score, 1.0, "score should be at most 1.0")

		t.Logf("Answer (bare object normalized): %q (score: %.4f)", answer.Answer, answer.Score)
	})

	// Test 2: With top_k=1 - API returns bare object, SDK normalizes to slice
	t.Run("with_topk_1_bare_object", func(t *testing.T) {
		topK := 1
		resp, err := client.AnswerQuestion(
			QuestionAnsweringRequest{
				Input: QuestionAnsweringInput{
					Question: question,
					Context:  context,
				},
				Parameters: &QuestionAnsweringParameters{
					TopK: &topK,
				},
			},
		)

		require.NoError(t, err, "question answering with top_k=1 should succeed")
		require.NotNil(t, resp, "response should not be nil")
		require.Len(t, resp, 1, "should have exactly 1 answer")

		answer := resp[0]
		require.NotEmpty(t, answer.Answer, "answer should not be empty")
		require.GreaterOrEqual(t, answer.Score, 0.0, "score should be non-negative")
		require.LessOrEqual(t, answer.Score, 1.0, "score should be at most 1.0")

		t.Logf("Answer (top_k=1, bare object normalized): %q (score: %.4f)", answer.Answer, answer.Score)
	})

	// Test 3: With top_k=3 - API returns array, SDK decodes as-is
	t.Run("with_topk_3_array", func(t *testing.T) {
		topK := 3
		resp, err := client.AnswerQuestion(
			QuestionAnsweringRequest{
				Input: QuestionAnsweringInput{
					Question: question,
					Context:  context,
				},
				Parameters: &QuestionAnsweringParameters{
					TopK: &topK,
				},
			},
		)

		require.NoError(t, err, "question answering with top_k=3 should succeed")
		require.NotNil(t, resp, "response should not be nil")
		require.NotEmpty(t, resp, "response should have answers")
		require.LessOrEqual(t, len(resp), topK, "should have at most top_k answers")

		for i, answer := range resp {
			require.NotEmpty(t, answer.Answer, "answer should not be empty")
			require.GreaterOrEqual(t, answer.Score, 0.0, "score should be non-negative")
			require.LessOrEqual(t, answer.Score, 1.0, "score should be at most 1.0")

			t.Logf("Answer %d (top_k=3, array): %q (score: %.4f)", i, answer.Answer, answer.Score)
		}

		// Verify answers are ordered by score (descending)
		for i := range len(resp) - 1 {
			require.GreaterOrEqual(t, resp[i].Score, resp[i+1].Score, "answers should be ordered by score (descending)")
		}
	})
}

// TestQuestionAnswering_WithParameters tests question answering with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestQuestionAnswering_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "deepset/roberta-base-squad2"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	maxAnswerLen := 5
	resp, err := client.AnswerQuestion(
		QuestionAnsweringRequest{
			Input: QuestionAnsweringInput{
				Question: "What is the capital of France?",
				Context:  "France is a country in Europe. Its capital is Paris. It is known for the Eiffel Tower.",
			},
			Parameters: &QuestionAnsweringParameters{
				MaxAnswerLen: &maxAnswerLen,
			},
		},
	)

	require.NoError(t, err, "question answering with parameters should succeed")
	require.NotNil(t, resp, "response should not be nil")
	require.NotEmpty(t, resp, "response should have answers")

	for _, answer := range resp {
		require.LessOrEqual(t, len(answer.Answer), maxAnswerLen, "answer length should respect max_answer_len")
	}
}

// TestQuestionAnswering_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestQuestionAnswering_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("deepset/roberta-base-squad2"),
		WithContext(ctx),
	)

	resp, err := client.AnswerQuestion(
		QuestionAnsweringRequest{
			Input: QuestionAnsweringInput{
				Question: "What is the capital of France?",
				Context:  "France is a country in Europe.",
			},
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
	require.Nil(t, resp, "response should be nil for cancelled context")
}
