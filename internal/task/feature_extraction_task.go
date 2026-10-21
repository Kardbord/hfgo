package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// ExtractPooledFeatures sends a feature extraction request for a single input.
func ExtractPooledFeatures(
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtractionPooledEmbedding, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionPooledProvider](opts.Provider)
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

	return doInference(opts, endpoint, prov.FeatureExtractionPooledCodec(), req)
}

// ExtractTokenFeatures sends a feature extraction request for a single input
// and returns token-level embeddings.
func ExtractTokenFeatures(
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtractionTokenEmbeddings, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionTokenProvider](opts.Provider)
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

	return doInference(opts, endpoint, prov.FeatureExtractionTokenCodec(), req)
}

// ExtractPooledFeaturesBatch sends a feature extraction request for a batch of
// inputs.
func ExtractPooledFeaturesBatch(
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionPooledBatchProvider](
		opts.Provider,
	)
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

	return doInference(opts, endpoint, prov.FeatureExtractionPooledBatchCodec(), req)
}

// ExtractTokenFeaturesBatch sends a feature extraction request for a batch of
// inputs and returns token-level embeddings.
func ExtractTokenFeaturesBatch(
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtractionTokenEmbeddings, error) {
	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionTokenBatchProvider](
		opts.Provider,
	)
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

	return doInference(opts, endpoint, prov.FeatureExtractionTokenBatchCodec(), req)
}
