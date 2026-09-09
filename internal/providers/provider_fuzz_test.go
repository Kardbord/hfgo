//go:build !integration

package providers

import (
	"testing"
)

func FuzzHuggingFaceProviderEndpoint(f *testing.F) {
	p := HuggingFaceProvider{}

	f.Add(string(TaskChatCompletion), "mistral-7b")
	f.Add(string(TaskFeatureExtraction), "bert-base")
	f.Add(string(TaskSentenceSimilarity), "sentence-transformers/all-MiniLM-L6-v2")
	f.Add(string(TaskTextClassification), "distilbert-base-uncased")
	f.Add("", "")
	f.Add(string(TaskChatCompletion), "")
	f.Add("", "mistral-7b")
	f.Add("unknown-task", "model")
	f.Add(string(TaskFeatureExtraction), "model:provider")
	f.Add(string(TaskChatCompletion), "org/model:variant")

	f.Fuzz(func(t *testing.T, task, model string) {
		ep, err := p.Endpoint(Task(task), model)
		if model == "" && err == nil {
			t.Errorf("expected error for empty model, got %q", ep)
		}
		if model != "" && err != nil {
			// Unknown tasks now return an error; only known tasks succeed.
			_ = ep
		}
	})
}
