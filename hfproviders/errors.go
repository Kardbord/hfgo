package hfproviders

import (
	"fmt"

	"github.com/Kardbord/hfgo/v4/hferrors"
)

// errNilResolver builds the configuration error returned when a WaveSpeed
// mixin or provider is used without a configured [ModelResolver]. A
// constructor is used instead of a package-global error value so callers
// always receive a fresh error they can wrap freely.
func errNilResolver(provider string) *hferrors.SDKError {
	return &hferrors.SDKError{
		Kind:    hferrors.SDKErrorKindConfiguration,
		Message: fmt.Sprintf("provider %s: resolver must not be nil", provider),
		Err:     nil,
	}
}
