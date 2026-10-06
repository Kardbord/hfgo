package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ExtractFeatures sends a feature extraction request for a single input.
func ExtractFeatures(
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtraction, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.FeatureExtractionCodec(), req)
}

// ExtractFeaturesBatch sends a feature extraction request for a batch of inputs.
func ExtractFeaturesBatch(
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtraction, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionBatchProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionBatchEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.FeatureExtractionBatchCodec(), req)
}
