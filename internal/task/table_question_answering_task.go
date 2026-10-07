package task

import (
	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// AnswerTableQuestion sends a table question answering request and returns the answer.
func AnswerTableQuestion(
	opts hfopts.Options,
	req hftypes.TableQuestionAnsweringRequest,
) (hftypes.TableQuestionAnswer, error) {
	if err := validateDispatch(opts); err != nil {
		return hftypes.TableQuestionAnswer{}, err
	}

	prov, err := hfproviders.AsProvider[hfproviders.TableQuestionAnsweringProvider](opts.Provider)
	if err != nil {
		return hftypes.TableQuestionAnswer{}, err
	}

	endpoint, err := prov.TableQuestionAnsweringEndpoint(hfproviders.EndpointParams{
		Context: opts.Context(),
		Model:   opts.Model,
	})
	if err != nil {
		return hftypes.TableQuestionAnswer{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "error computing endpoint: " + err.Error(),
			Err:     err,
		}
	}

	return doInference(opts, endpoint, prov.TableQuestionAnsweringCodec(), req)
}
