//go:build !integration

package hfproviders

import (
	"errors"
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

func FuzzHuggingFaceProviderEndpoint(f *testing.F) {
	p := HuggingFaceProvider{}

	seeds := []string{
		"mistral-7b",
		"bert-base",
		"sentence-transformers/all-MiniLM-L6-v2",
		"distilbert-base-uncased",
		"facebook/bart-large-mnli",
		"dslim/bert-base-NER",
		"deepset/roberta-base-squad2",
		"google/tapas-base-finetuned-wtq",
		"google-bert/bert-base-uncased",
		"facebook/bart-large-cnn",
		"t5-base",
		"stabilityai/stable-diffusion-xl-base-1.0",
		"meta-llama/Llama-3.1-8B-Instruct",
		"MIT/ast-finetuned-audioset-10-10-0.4593",
		"openai/whisper-large-v3",
		"google/vit-base-patch16-224",
		"facebook/detr-resnet-50-panoptic",
		"impira/layoutlm-document-qa",
		"Salesforce/blip-image-captioning-base",
		"facebook/detr-resnet-50",
		"speechbrain/sepformer-wham",
		"openai/clip-vit-base-patch32",
		"timbrooks/instruct-pix2pix",
		"scikit-learn/fish-weight",
		"microsoft/speecht5_tts",
		"dandelin/vilt-b32-finetuned-vqa",
		"facebook/musicgen-small",
		"damo-vilab/text-to-video-ms-1.7b",
		"stabilityai/stable-video-diffusion-img2vid-xt",
		"black-forest-labs/FLUX.1-schnell",
		"tencent/HunyuanVideo",
		"model:provider",
		"org/model:variant",
		"",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, model string) {
		params := EndpointParams{Model: model}

		check := func(name string, call func(EndpointParams) (Endpoint, error), want func(string) string, allowEmpty bool) {
			t.Helper()

			ep, err := call(params)
			if model == "" && !allowEmpty {
				if err == nil {
					t.Errorf("%s(%q): expected error for empty model, got %+v", name, model, ep)

					return
				}

				var sdkErr *hferrors.SDKError
				if !errors.As(err, &sdkErr) || sdkErr.Kind != hferrors.SDKErrorKindConfiguration {
					t.Errorf("%s(%q): expected Configuration SDKError, got %v", name, model, err)
				}

				return
			}

			if err != nil {
				t.Errorf("%s(%q) returned unexpected error: %v", name, model, err)

				return
			}

			if ep.Method != http.MethodPost {
				t.Errorf("%s(%q) method = %q, want %q", name, model, ep.Method, http.MethodPost)
			}

			if expected := want(model); ep.Path != expected {
				t.Errorf("%s(%q) path = %q, want %q", name, model, ep.Path, expected)
			}
		}

		// ChatEndpoint is model-independent: it must never reject an empty
		// model and must always return the fixed path.
		check("ChatEndpoint", p.ChatEndpoint, func(string) string {
			return "v1/chat/completions"
		}, true)

		// FeatureExtractionEndpoint embeds the model under a pipeline path.
		check("FeatureExtractionEndpoint", p.FeatureExtractionEndpoint, func(m string) string {
			return hfModelPrefix + m + "/pipeline/feature-extraction"
		}, false)

		// The remaining model-dependent endpoints embed the model directly
		// under the prefix: hf-inference/models/{model}.
		modelPath := func(m string) string { return hfModelPrefix + m }
		for _, tc := range []struct {
			name string
			call func(EndpointParams) (Endpoint, error)
		}{
			{"DetectObjectsEndpoint", p.DetectObjectsEndpoint},
			{"ImageClassificationEndpoint", p.ImageClassificationEndpoint},
			{"ImageSegmentationEndpoint", p.ImageSegmentationEndpoint},
			{"TextClassificationEndpoint", p.TextClassificationEndpoint},
			{"ZeroShotTextClassificationEndpoint", p.ZeroShotTextClassificationEndpoint},
			{"TokenClassificationEndpoint", p.TokenClassificationEndpoint},
			{"QuestionAnsweringEndpoint", p.QuestionAnsweringEndpoint},
			{"TableQuestionAnsweringEndpoint", p.TableQuestionAnsweringEndpoint},
			{"TextToImageEndpoint", p.TextToImageEndpoint},
			{"FillMaskEndpoint", p.FillMaskEndpoint},
			{"SummarizationEndpoint", p.SummarizationEndpoint},
			{"TranslationEndpoint", p.TranslationEndpoint},
		} {
			check(tc.name, tc.call, modelPath, false)
		}
	})
}
