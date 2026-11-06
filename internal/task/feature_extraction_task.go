package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// ExtractPooledFeatures sends a feature extraction request for a single input.
func ExtractPooledFeatures(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtractionPooledEmbedding, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionPooledProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.FeatureExtractionPooledCodec(), req)
}

// ExtractTokenFeatures sends a feature extraction request for a single input
// and returns token-level embeddings.
func ExtractTokenFeatures(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.FeatureExtractionRequest,
) (hftypes.FeatureExtractionTokenEmbeddings, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionTokenProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.FeatureExtractionTokenCodec(), req)
}

// ExtractPooledFeaturesBatch sends a feature extraction request for a batch of
// inputs.
func ExtractPooledFeaturesBatch(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtractionPooledEmbedding, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionPooledBatchProvider](
		opts.Provider,
	)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionBatchEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.FeatureExtractionPooledBatchCodec(), req)
}

// ExtractTokenFeaturesBatch sends a feature extraction request for a batch of
// inputs and returns token-level embeddings.
func ExtractTokenFeaturesBatch(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.FeatureExtractionBatchRequest,
) ([]hftypes.FeatureExtractionTokenEmbeddings, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.FeatureExtractionTokenBatchProvider](
		opts.Provider,
	)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.FeatureExtractionBatchEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.FeatureExtractionTokenBatchCodec(), req)
}
