//go:build integration

package hfgo

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestTableQuestionAnswering_LiveAPI tests a basic table question answering request against the live HF API.
// This test requires the HF_TOKEN environment variable to be set.
func TestTableQuestionAnswering_LiveAPI(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google/tapas-base-finetuned-wtq"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	resp, err := client.AnswerTableQuestion(
		TableQuestionAnsweringRequest{
			Input: TableQuestionAnsweringInput{
				Question: "How old is Bob?",
				Table: map[string][]string{
					"Name": {"Alice", "Bob", "Carol"},
					"Age":  {"25", "30", "35"},
				},
			},
		},
	)

	require.NoError(t, err, "table question answering should succeed")
	require.NotEmpty(t, resp.Answer, "answer should not be empty")
	require.NotEmpty(t, resp.Cells, "cells should not be empty")
	require.NotEmpty(t, resp.Coordinates, "coordinates should not be empty")

	t.Logf("Answer: %q (cells: %v, coordinates: %v)", resp.Answer, resp.Cells, resp.Coordinates)
}

// TestTableQuestionAnswering_WithParameters tests table question answering with various parameters.
// This test requires the HF_TOKEN environment variable to be set.
func TestTableQuestionAnswering_WithParameters(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	const model = "google/tapas-base-finetuned-wtq"

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel(model),
		WithContext(ctx),
	)

	padding := TableQuestionAnsweringPaddingMaxLength
	sequential := false
	truncation := true
	resp, err := client.AnswerTableQuestion(
		TableQuestionAnsweringRequest{
			Input: TableQuestionAnsweringInput{
				Question: "What is the most expensive product?",
				Table: map[string][]string{
					"Product": {"Laptop", "Phone", "Tablet"},
					"Price":   {"1200", "800", "500"},
				},
			},
			Parameters: &TableQuestionAnsweringParameters{
				Padding:    &padding,
				Sequential: &sequential,
				Truncation: &truncation,
			},
		},
	)

	require.NoError(t, err, "table question answering with parameters should succeed")
	require.NotEmpty(t, resp.Answer, "answer should not be empty")

	t.Logf("Answer: %q (cells: %v)", resp.Answer, resp.Cells)
}

// TestTableQuestionAnswering_ContextCancellation tests that context cancellation is respected.
// This test requires the HF_TOKEN environment variable to be set.
func TestTableQuestionAnswering_ContextCancellation(t *testing.T) {
	apiToken := os.Getenv("HF_TOKEN")
	require.NotEmpty(t, apiToken, "HF_TOKEN must be set")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := NewClient(
		WithToken(apiToken),
		WithModel("google/tapas-base-finetuned-wtq"),
		WithContext(ctx),
	)

	_, err := client.AnswerTableQuestion(
		TableQuestionAnsweringRequest{
			Input: TableQuestionAnsweringInput{
				Question: "How old is Bob?",
				Table: map[string][]string{
					"Name": {"Alice", "Bob", "Carol"},
					"Age":  {"25", "30", "35"},
				},
			},
		},
	)

	require.Error(t, err, "request with cancelled context should fail")
}
