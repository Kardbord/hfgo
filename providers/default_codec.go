package providers

// DefaultCodec is an identity codec that passes request and response bodies
// through unchanged. It is the baseline wire format used by the HuggingFace
// inference API and OpenAI-compatible endpoints. Providers that speak the
// same wire format for a given task can embed DefaultCodec to inherit the
// identity request and response transforms.
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
