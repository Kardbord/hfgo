package task

import (
	"context"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/utils"
)

// AnswerQuestion sends a question answering request and returns the answers.
func AnswerQuestion(
	ctx context.Context,
	opts hfopts.Options,
	req hftypes.QuestionAnsweringRequest,
) ([]hftypes.QuestionAnswer, error) {
	ctx = utils.NormalizeContext(ctx)

	if err := validateDispatch(opts); err != nil {
		return nil, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.QuestionAnsweringProvider](opts.Provider)
	if err != nil {
		return nil, err
	}

	endpoint, err := prov.QuestionAnsweringEndpoint(ctx, hfproviders.EndpointParams{
		Model: opts.Model,
	})
	if err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(ctx, opts, endpoint, prov.QuestionAnsweringCodec(), req)
}
