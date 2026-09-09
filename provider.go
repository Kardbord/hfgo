package hfgo

import "github.com/Kardbord/hfgo/v4/internal/providers"

// Provider knows how to construct API endpoints for a given task and model.
type Provider = providers.Provider

// Task identifies an inference task type.
type Task = providers.Task

const (
	// TaskTextClassification is the text classification task.
	TaskTextClassification = providers.TaskTextClassification
	// TaskZeroShotTextClassification is the zero-shot text classification task.
	TaskZeroShotTextClassification = providers.TaskZeroShotTextClassification
	// TaskTokenClassification is the token classification task.
	TaskTokenClassification = providers.TaskTokenClassification
	// TaskQuestionAnswering is the question answering task.
	TaskQuestionAnswering = providers.TaskQuestionAnswering
	// TaskTableQuestionAnswering is the table question answering task.
	TaskTableQuestionAnswering = providers.TaskTableQuestionAnswering
	// TaskFillMask is the fill mask task.
	TaskFillMask = providers.TaskFillMask
	// TaskSummarization is the summarization task.
	TaskSummarization = providers.TaskSummarization
	// TaskTranslation is the translation task.
	TaskTranslation = providers.TaskTranslation
	// TaskFeatureExtraction is the feature extraction task.
	TaskFeatureExtraction = providers.TaskFeatureExtraction
	// TaskSentenceSimilarity is the sentence similarity task.
	TaskSentenceSimilarity = providers.TaskSentenceSimilarity
	// TaskChatCompletion is the chat completion task.
	TaskChatCompletion = providers.TaskChatCompletion
)
