//go:build !integration

package providers

import (
	"testing"
)

func FuzzHuggingFaceProviderEndpoint(f *testing.F) {
	p := HuggingFaceProvider{}

	f.Add("chat-completion", "mistral-7b")
	f.Add("feature-extraction", "bert-base")
	f.Add("sentence-similarity", "sentence-transformers/all-MiniLM-L6-v2")
	f.Add("text-classification", "distilbert-base-uncased")
	f.Add("", "")
	f.Add("chat-completion", "")
	f.Add("", "mistral-7b")
	f.Add("unknown-task", "model")
	f.Add("feature-extraction", "model:provider")
	f.Add("chat-completion", "org/model:variant")

	f.Fuzz(func(t *testing.T, task, model string) {
		ep, err := p.Endpoint(task, model)
		if model == "" && err == nil {
			t.Errorf("expected error for empty model, got %q", ep)
		}
		if model != "" && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
}
