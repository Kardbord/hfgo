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
		"Name": {"Alice", "Bob", "Carol"},
		"Age":  {"25", "30", "35"},
		"City": {"New York", "London", "Paris"},
	}
	question := "How old is Bob?"

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Table:\n")
	PrintJSON(table)
	fmt.Println("...")

	answer, err := client.AnswerTableQuestion(
		hftypes.TableQuestionAnsweringRequest{
			Input: hftypes.TableQuestionAnsweringInput{
				Question: question,
				Table:    table,
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
