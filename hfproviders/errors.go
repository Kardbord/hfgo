package hfproviders

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// errNilResolver builds the configuration error returned when a provider or
// mixin requires a [ModelResolver] but none was configured — the async
// partner providers (WaveSpeed, fal.ai, Replicate) and the pipeline
// providers that map models to OpenAI-compatible endpoints (DeepInfra,
// Scaleway, Nscale). A constructor
// is used instead of a package-global error value so callers always receive
// a fresh error they can wrap freely.
func errNilResolver(provider string) *hferrors.SDKError {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindConfiguration,
		Message: fmt.Sprintf("provider %s: resolver must not be nil", provider),
		Err:     nil,
	}
}

// errUnexpectedContentType builds the serialization error every JSON-decoding
// codec returns when a response's Content-Type is not the expected
// application/json (or structured +json) media type. got is the raw header
// value, echoed for debugging.
func errUnexpectedContentType(got string) *hferrors.SDKError {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: "expected Content-Type application/json, got " + got,
		Err:     nil,
	}
}

// errUnexpectedMediaPrefix builds the serialization error surfaced when a
// media response's advertised or detected media type does not match the
// required family prefix (for example "image/"), so the codec cannot safely
// treat the body as that media family. got is the offending media type,
// echoed for debugging.
func errUnexpectedMediaPrefix(prefix, got string) *hferrors.SDKError {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindSerialization,
		Message: "expected a " + prefix + "* Content-Type, got " + got,
		Err:     nil,
	}
}
