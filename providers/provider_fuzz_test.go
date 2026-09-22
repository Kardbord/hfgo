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
	f.Add(string(TaskZeroShotTextClassification), "facebook/bart-large-mnli")
	f.Add(string(TaskTokenClassification), "dslim/bert-base-NER")
	f.Add(string(TaskQuestionAnswering), "deepset/roberta-base-squad2")
	f.Add(string(TaskTableQuestionAnswering), "google/tapas-base-finetuned-wtq")
	f.Add(string(TaskFillMask), "google-bert/bert-base-uncased")
	f.Add(string(TaskSummarization), "facebook/bart-large-cnn")
	f.Add(string(TaskTranslation), "t5-base")
	f.Add(string(TaskTextToImage), "stabilityai/stable-diffusion-xl-base-1.0")
	f.Add(string(TaskTextGeneration), "meta-llama/Llama-3.1-8B-Instruct")
	f.Add(string(TaskAudioClassification), "MIT/ast-finetuned-audioset-10-10-0.4593")
	f.Add(string(TaskAutomaticSpeechRecognition), "openai/whisper-large-v3")
	f.Add(string(TaskImageClassification), "google/vit-base-patch16-224")
	f.Add(string(TaskImageSegmentation), "facebook/detr-resnet-50-panoptic")
	f.Add(string(TaskDocumentQuestionAnswering), "impira/layoutlm-document-qa")
	f.Add(string(TaskImageToText), "Salesforce/blip-image-captioning-base")
	f.Add(string(TaskObjectDetection), "facebook/detr-resnet-50")
	f.Add(string(TaskAudioToAudio), "speechbrain/sepformer-wham")
	f.Add(string(TaskZeroShotImageClassification), "openai/clip-vit-base-patch32")
	f.Add(string(TaskImageToImage), "timbrooks/instruct-pix2pix")
	f.Add(string(TaskTabularClassification), "scikit-learn/fish-weight")
	f.Add(string(TaskTextToSpeech), "microsoft/speecht5_tts")
	f.Add(string(TaskVisualQuestionAnswering), "dandelin/vilt-b32-finetuned-vqa")
	f.Add(string(TaskTabularRegression), "scikit-learn/fish-weight")
	f.Add(string(TaskTextToAudio), "facebook/musicgen-small")
	f.Add(string(TaskTextToVideo), "damo-vilab/text-to-video-ms-1.7b")
	f.Add(string(TaskImageToVideo), "stabilityai/stable-video-diffusion-img2vid-xt")
	f.Add(string(TaskImageTextToImage), "black-forest-labs/FLUX.1-schnell")
	f.Add(string(TaskImageTextToVideo), "tencent/HunyuanVideo")
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
		if task == "" && err == nil {
			t.Errorf("expected error for empty task, got %q", ep)
		}
	})
}
