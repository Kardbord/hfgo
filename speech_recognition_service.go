package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// speechRecognitionService implements automatic speech recognition calls
// using the configured request options.
type speechRecognitionService struct {
	opts request.Options
}

// newSpeechRecognitionService builds a speech recognition service with a
// snapshot of the provided options.
func newSpeechRecognitionService(opts request.Options) speechRecognitionService {
	return speechRecognitionService{opts: opts}
}

// recognize sends a speech recognition request for a single input and
// returns the output.
func (s speechRecognitionService) recognize(
	req SpeechRecognitionRequest,
	opts ...Option,
) (SpeechRecognition, error) {
	return doModelInference[SpeechRecognitionRequest, SpeechRecognition](
		s.opts.With(opts...),
		TaskAutomaticSpeechRecognition,
		req,
	)
}

// recognizeBatch sends a speech recognition request for a batch of inputs
// and returns a flat list of outputs, one for each input in the batch.
func (s speechRecognitionService) recognizeBatch(
	req SpeechRecognitionBatchRequest,
	opts ...Option,
) ([]SpeechRecognition, error) {
	return doModelInference[SpeechRecognitionBatchRequest, []SpeechRecognition](
		s.opts.With(opts...),
		TaskAutomaticSpeechRecognition,
		req,
	)
}
