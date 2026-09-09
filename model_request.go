package hfgo

import (
	"net/http"

	"github.com/Kardbord/hfgo/v4/internal/request"
)

// doModelInference posts req to the inference model endpoint for the model
// configured in opts, returning the typed response. It returns a configuration
// SDK error when no model is set. task names the inference task in that error
// message, e.g. "fill mask" or "summarization".
//
// It is intended for pointer or slice response types: on the configuration
// error path it returns the typed zero value of Resp, which is nil for slices
// and pointers and so also signals "no result" to callers.
func doModelInference[Req, Resp any](opts request.Options, task string, req Req) (Resp, error) {
	var zero Resp

	if opts.Model == "" {
		return zero, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "the model option must be set for " + task + " to succeed",
			Err:     nil,
		}
	}

	if opts.Provider == nil {
		return zero, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "provider must not be nil",
			Err:     nil,
		}
	}

	endpoint, err := opts.Provider.Endpoint(task, opts.Model)
	if err != nil {
		return zero, err
	}

	return request.DoJSON[Req, Resp](
		opts,
		http.MethodPost,
		endpoint,
		req,
	)
}
