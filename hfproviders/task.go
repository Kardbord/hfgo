package hfproviders

// Task identifies an upstream Hugging Face task whose provider mapping is
// being resolved. Provider mappings are task-specific on the Hub, so the task
// is part of a resolver lookup key. The constants below name the Hub task
// identifiers for the tasks the SDK's providers serve; other identifiers
// remain expressible by converting a string, e.g. Task("custom-task").
type Task string

const (
	// TaskTextGeneration is the text-generation task.
	TaskTextGeneration Task = "text-generation"
	// TaskConversational is the conversational (chat) task. Chat performs
	// provider mapping server-side, so it normally does not use a resolver.
	TaskConversational Task = "conversational"
	// TaskImageToImage is the image-to-image task.
	TaskImageToImage Task = "image-to-image"
	// TaskTextToImage is the text-to-image task.
	TaskTextToImage Task = "text-to-image"
	// TaskTextToVideo is the text-to-video task.
	TaskTextToVideo Task = "text-to-video"
	// TaskAutomaticSpeechRecognition is the speech-recognition task.
	TaskAutomaticSpeechRecognition Task = "automatic-speech-recognition"
	// TaskFeatureExtraction is the feature-extraction task.
	TaskFeatureExtraction Task = "feature-extraction"
	// TaskFillMask is the fill-mask task.
	TaskFillMask Task = "fill-mask"
	// TaskImageClassification is the image-classification task.
	TaskImageClassification Task = "image-classification"
	// TaskObjectDetection is the object-detection task.
	TaskObjectDetection Task = "object-detection"
	// TaskImageSegmentation is the image-segmentation task.
	TaskImageSegmentation Task = "image-segmentation"
	// TaskQuestionAnswering is the question-answering task.
	TaskQuestionAnswering Task = "question-answering"
	// TaskSummarization is the summarization task.
	TaskSummarization Task = "summarization"
	// TaskTableQuestionAnswering is the table-question-answering task.
	TaskTableQuestionAnswering Task = "table-question-answering"
	// TaskTextClassification is the text-classification task.
	TaskTextClassification Task = "text-classification"
	// TaskTokenClassification is the token-classification task.
	TaskTokenClassification Task = "token-classification"
	// TaskTranslation is the translation task.
	TaskTranslation Task = "translation"
	// TaskZeroShotTextClassification is the zero-shot text-classification task.
	TaskZeroShotTextClassification Task = "zero-shot-classification"
)
