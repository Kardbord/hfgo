package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/Kardbord/hfgo/v4"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

func main() {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		log.Fatal("HF_TOKEN environment variable is not set")
	}

	client := hfgo.NewClient(
		hfopts.WithToken(token),
		hfopts.WithModel("google/tapas-base-finetuned-wtq"),
	)

	table := map[string][]string{
		"Product": {"Laptop", "Phone", "Tablet"},
		"Price":   {"1200", "800", "500"},
	}
	question := "What is the most expensive product?"

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Table:\n")
	PrintJSON(table)
	fmt.Println("...")

	padding := hftypes.TableQuestionAnsweringPaddingMaxLength
	sequential := false
	truncation := true
	answer, err := client.AnswerTableQuestion(
		hftypes.TableQuestionAnsweringRequest{
			Input: hftypes.TableQuestionAnsweringInput{
				Question: question,
				Table:    table,
			},
			Parameters: &hftypes.TableQuestionAnsweringParameters{
				Padding:    &padding,
				Sequential: &sequential,
				Truncation: &truncation,
			},
		},
	)
	if err != nil {
		log.Fatalf("error running table question answering: %v\n", err)
	}

	fmt.Println("Results:")
	PrintJSON(answer)
}

func PrintJSON[T any](v T) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("error printing JSON: %v\n", err)
	}

	fmt.Println(string(b))
}
