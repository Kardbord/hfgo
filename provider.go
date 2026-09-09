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
	// TaskTextToImage is the text to image task.
	TaskTextToImage = providers.TaskTextToImage
	// TaskTextGeneration is the text generation task.
	TaskTextGeneration = providers.TaskTextGeneration
	// TaskAudioClassification is the audio classification task.
	TaskAudioClassification = providers.TaskAudioClassification
	// TaskAutomaticSpeechRecognition is the automatic speech recognition task.
	TaskAutomaticSpeechRecognition = providers.TaskAutomaticSpeechRecognition
	// TaskImageClassification is the image classification task.
	TaskImageClassification = providers.TaskImageClassification
	// TaskImageSegmentation is the image segmentation task.
	TaskImageSegmentation = providers.TaskImageSegmentation
	// TaskDocumentQuestionAnswering is the document question answering task.
	TaskDocumentQuestionAnswering = providers.TaskDocumentQuestionAnswering
	// TaskImageToText is the image to text task.
	TaskImageToText = providers.TaskImageToText
	// TaskObjectDetection is the object detection task.
	TaskObjectDetection = providers.TaskObjectDetection
	// TaskAudioToAudio is the audio to audio task.
	TaskAudioToAudio = providers.TaskAudioToAudio
	// TaskZeroShotImageClassification is the zero-shot image classification task.
	TaskZeroShotImageClassification = providers.TaskZeroShotImageClassification
	// TaskImageToImage is the image to image task.
	TaskImageToImage = providers.TaskImageToImage
	// TaskTabularClassification is the tabular classification task.
	TaskTabularClassification = providers.TaskTabularClassification
	// TaskTextToSpeech is the text to speech task.
	TaskTextToSpeech = providers.TaskTextToSpeech
	// TaskVisualQuestionAnswering is the visual question answering task.
	TaskVisualQuestionAnswering = providers.TaskVisualQuestionAnswering
	// TaskTabularRegression is the tabular regression task.
	TaskTabularRegression = providers.TaskTabularRegression
	// TaskTextToAudio is the text to audio task.
	TaskTextToAudio = providers.TaskTextToAudio
	// TaskTextToVideo is the text to video task.
	TaskTextToVideo = providers.TaskTextToVideo
	// TaskImageToVideo is the image to video task.
	TaskImageToVideo = providers.TaskImageToVideo
	// TaskImageTextToImage is the image text to image task.
	TaskImageTextToImage = providers.TaskImageTextToImage
	// TaskImageTextToVideo is the image text to video task.
	TaskImageTextToVideo = providers.TaskImageTextToVideo
)
