package providers

// Task identifies an inference task type.
type Task string

const (
	// TaskTextClassification is the text classification task.
	TaskTextClassification Task = "text-classification"
	// TaskZeroShotTextClassification is the zero-shot text classification task.
	TaskZeroShotTextClassification Task = "zero-shot-text-classification"
	// TaskTokenClassification is the token classification task.
	TaskTokenClassification Task = "token-classification"
	// TaskQuestionAnswering is the question answering task.
	TaskQuestionAnswering Task = "question-answering"
	// TaskTableQuestionAnswering is the table question answering task.
	TaskTableQuestionAnswering Task = "table-question-answering"
	// TaskFillMask is the fill mask task.
	TaskFillMask Task = "fill-mask"
	// TaskSummarization is the summarization task.
	TaskSummarization Task = "summarization"
	// TaskTranslation is the translation task.
	TaskTranslation Task = "translation"
	// TaskFeatureExtraction is the feature extraction task.
	TaskFeatureExtraction Task = "feature-extraction"
	// TaskSentenceSimilarity is the sentence similarity task.
	TaskSentenceSimilarity Task = "sentence-similarity"
	// TaskChatCompletion is the chat completion task.
	TaskChatCompletion Task = "chat-completion"
)
