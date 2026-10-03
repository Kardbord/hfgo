// Package hfproviders defines the Provider interface and built-in provider implementations.
package hfproviders

// Provider knows how to construct API endpoints and transform between HF-spec
// and provider-spec wire formats for a given task and model. Implementations
// must be safe for concurrent use.
type Provider interface {
	// Endpoint returns the API endpoint path for the given task and model.
	// It returns an error if the model is invalid or the task is unsupported.
	Endpoint(task Task, model string) (string, error)

	// ProviderSuffix returns the provider suffix appended to model IDs
	// for routing in OpenAI-compatible endpoints. An empty string means
	// no suffix is appended (e.g. for the default HuggingFace provider).
	ProviderSuffix() string
}

// ChatCodecProvider is a Provider that supplies a Codec for the Chat task.
type ChatCodecProvider interface {
	Provider
	ChatCodec() ChatCodec
}

// ChatStreamCodecProvider is a Provider that supplies a Codec for the ChatStream task.
type ChatStreamCodecProvider interface {
	Provider
	ChatStreamCodec() ChatStreamCodec
}

// FeatureExtractionBatchCodecProvider is a Provider that supplies a Codec for the
// FeatureExtractBatch task.
type FeatureExtractionBatchCodecProvider interface {
	Provider
	FeatureExtractionBatchCodec() FeatureExtractionBatchCodec
}

// FeatureExtractionCodecProvider is a Provider that supplies a Codec for the
// FeatureExtract task.
type FeatureExtractionCodecProvider interface {
	Provider
	FeatureExtractionCodec() FeatureExtractionCodec
}

// FillMaskBatchCodecProvider is a Provider that supplies a Codec for the FillMaskBatch task.
type FillMaskBatchCodecProvider interface {
	Provider
	FillMaskBatchCodec() FillMaskBatchCodec
}

// FillMaskCodecProvider is a Provider that supplies a Codec for the FillMask task.
type FillMaskCodecProvider interface {
	Provider
	FillMaskCodec() FillMaskCodec
}

// QuestionAnsweringCodecProvider is a Provider that supplies a Codec for the
// AnswerQuestion task.
type QuestionAnsweringCodecProvider interface {
	Provider
	QuestionAnsweringCodec() QuestionAnsweringCodec
}

// SummarizationBatchCodecProvider is a Provider that supplies a Codec for the
// SummarizeBatch task.
type SummarizationBatchCodecProvider interface {
	Provider
	SummarizationBatchCodec() SummarizationBatchCodec
}

// SummarizationCodecProvider is a Provider that supplies a Codec for the Summarize task.
type SummarizationCodecProvider interface {
	Provider
	SummarizationCodec() SummarizationCodec
}

// TableQuestionAnsweringCodecProvider is a Provider that supplies a Codec for the
// AnswerTableQuestion task.
type TableQuestionAnsweringCodecProvider interface {
	Provider
	TableQuestionAnsweringCodec() TableQuestionAnsweringCodec
}

// TextClassificationBatchCodecProvider is a Provider that supplies a Codec for the
// ClassifyTextBatch task.
type TextClassificationBatchCodecProvider interface {
	Provider
	TextClassificationBatchCodec() TextClassificationBatchCodec
}

// TextClassificationCodecProvider is a Provider that supplies a Codec for the
// ClassifyText task.
type TextClassificationCodecProvider interface {
	Provider
	TextClassificationCodec() TextClassificationCodec
}

// TokenClassificationBatchCodecProvider is a Provider that supplies a Codec for the
// ClassifyTokensBatch task.
type TokenClassificationBatchCodecProvider interface {
	Provider
	TokenClassificationBatchCodec() TokenClassificationBatchCodec
}

// TokenClassificationCodecProvider is a Provider that supplies a Codec for the
// ClassifyTokens task.
type TokenClassificationCodecProvider interface {
	Provider
	TokenClassificationCodec() TokenClassificationCodec
}

// TranslationBatchCodecProvider is a Provider that supplies a Codec for the
// TranslateBatch task.
type TranslationBatchCodecProvider interface {
	Provider
	TranslationBatchCodec() TranslationBatchCodec
}

// TranslationCodecProvider is a Provider that supplies a Codec for the Translate task.
type TranslationCodecProvider interface {
	Provider
	TranslationCodec() TranslationCodec
}

// ZeroShotTextClassificationBatchCodecProvider is a Provider that supplies a Codec
// for the ZeroShotClassifyTextBatch task.
type ZeroShotTextClassificationBatchCodecProvider interface {
	Provider
	ZeroShotTextClassificationBatchCodec() ZeroShotTextClassificationBatchCodec
}

// ZeroShotTextClassificationCodecProvider is a Provider that supplies a Codec for
// the ZeroShotClassifyText task.
type ZeroShotTextClassificationCodecProvider interface {
	Provider
	ZeroShotTextClassificationCodec() ZeroShotTextClassificationCodec
}
