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
	// TaskTextToImage is the text to image task.
	TaskTextToImage Task = "text-to-image"
	// TaskTextGeneration is the text generation task.
	TaskTextGeneration Task = "text-generation"
	// TaskAudioClassification is the audio classification task.
	TaskAudioClassification Task = "audio-classification"
	// TaskAutomaticSpeechRecognition is the automatic speech recognition task.
	TaskAutomaticSpeechRecognition Task = "automatic-speech-recognition"
	// TaskImageClassification is the image classification task.
	TaskImageClassification Task = "image-classification"
	// TaskImageSegmentation is the image segmentation task.
	TaskImageSegmentation Task = "image-segmentation"
	// TaskDocumentQuestionAnswering is the document question answering task.
	TaskDocumentQuestionAnswering Task = "document-question-answering"
	// TaskImageToText is the image to text task.
	TaskImageToText Task = "image-to-text"
	// TaskObjectDetection is the object detection task.
	TaskObjectDetection Task = "object-detection"
	// TaskAudioToAudio is the audio to audio task.
	TaskAudioToAudio Task = "audio-to-audio"
	// TaskZeroShotImageClassification is the zero-shot image classification task.
	TaskZeroShotImageClassification Task = "zero-shot-image-classification"
	// TaskImageToImage is the image to image task.
	TaskImageToImage Task = "image-to-image"
	// TaskTabularClassification is the tabular classification task.
	TaskTabularClassification Task = "tabular-classification"
	// TaskTextToSpeech is the text to speech task.
	TaskTextToSpeech Task = "text-to-speech"
	// TaskVisualQuestionAnswering is the visual question answering task.
	TaskVisualQuestionAnswering Task = "visual-question-answering"
	// TaskTabularRegression is the tabular regression task.
	TaskTabularRegression Task = "tabular-regression"
	// TaskTextToAudio is the text to audio task.
	TaskTextToAudio Task = "text-to-audio"
	// TaskTextToVideo is the text to video task.
	TaskTextToVideo Task = "text-to-video"
	// TaskImageToVideo is the image to video task.
	TaskImageToVideo Task = "image-to-video"
	// TaskImageTextToImage is the image text to image task.
	TaskImageTextToImage Task = "image-text-to-image"
	// TaskImageTextToVideo is the image text to video task.
	TaskImageTextToVideo Task = "image-text-to-video"
)
