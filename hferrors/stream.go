package hferrors

// EndOfStreamError is a control signal returned by decoders, such as
// hfproviders.Codec implementations, to end a stream early. The stream
// pipeline discards the current payload, closes the underlying stream, and
// reports io.EOF to the consumer. Use it when a provider's framing carries
// the end of the stream in-band and no transport-level end signal exists
// (for example, an Anthropic-style "message_stop" event).
//
// Return the zero value, EndOfStreamError{}. Wrapped forms (fmt.Errorf with
// %w) and the pointer form are recognized as well. The signal only has
// meaning during streaming; unary pipelines treat any non-nil error as a
// failure and return it as-is.
type EndOfStreamError struct{}

// Error implements the error interface.
func (EndOfStreamError) Error() string {
	return "stream: end of stream signaled by decoder"
}

// Is reports whether target denotes this control signal in value or pointer
// form, so errors.Is matches both spellings.
func (EndOfStreamError) Is(target error) bool {
	_, isValue := target.(EndOfStreamError)
	_, isPointer := target.(*EndOfStreamError)

	return isValue || isPointer
}

// SkipEventError is a control signal returned by decoders, such as
// hfproviders.Codec implementations, to suppress a single streaming frame
// without delivering a value to the consumer. Use it for frames that carry
// no decodable state of their own (for example, an Anthropic-style
// "content_block_start" event) and that would otherwise surface as phantom
// events.
//
// Return the zero value, SkipEventError{}. Wrapped forms (fmt.Errorf with
// %w) and the pointer form are recognized as well. The signal only has
// meaning during streaming; unary pipelines treat any non-nil error as a
// failure and return it as-is.
type SkipEventError struct{}

// Error implements the error interface.
func (SkipEventError) Error() string {
	return "stream: event skipped by decoder"
}

// Is reports whether target denotes this control signal in value or pointer
// form, so errors.Is matches both spellings.
func (SkipEventError) Is(target error) bool {
	_, isValue := target.(SkipEventError)
	_, isPointer := target.(*SkipEventError)

	return isValue || isPointer
}
