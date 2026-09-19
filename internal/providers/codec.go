package providers

// Codec transforms between HF-spec and provider-spec wire formats for a task.
// Implementations transform request bodies before sending and response bodies
// after receiving, adapting the HF-format request/response to a provider's
// native wire format. They must be safe for concurrent use.
type Codec interface {
	// EncodeRequest transforms an HF-format request body into a provider-format
	// request body. hfContentType is the Content-Type of the HF-format body
	// (e.g. "application/json", "image/png"). Returns the transformed body and
	// the Content-Type to use when sending to the provider.
	EncodeRequest(task Task, hfBody []byte, hfContentType string) (
		providerBody []byte, providerContentType string, err error,
	)

	// DecodeResponse transforms a provider-format response body into an
	// HF-format response body. providerContentType is the Content-Type of the
	// provider's response. Returns the transformed body and the Content-Type
	// of the transformed body.
	DecodeResponse(task Task, providerBody []byte, providerContentType string) (
		hfBody []byte, hfContentType string, err error,
	)
}

// DefaultCodec is an identity codec that passes request and response bodies
// through unchanged. It is the baseline wire format used by the HuggingFace
// inference API and OpenAI-compatible endpoints. Providers that speak the
// same wire format for a given task can use DefaultCodec for that task.
type DefaultCodec struct{}

// EncodeRequest returns the input unchanged.
func (DefaultCodec) EncodeRequest(
	_ Task, body []byte, ct string,
) (providerBody []byte, providerContentType string, err error) {
	return body, ct, nil
}

// DecodeResponse returns the input unchanged.
func (DefaultCodec) DecodeResponse(
	_ Task, body []byte, ct string,
) (hfBody []byte, hfContentType string, err error) {
	return body, ct, nil
}
