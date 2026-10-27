// Package transport bridges the public hfopts.Options type to the internal
// request pipeline without exposing internal request types in the public API.
package transport

import (
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// Params projects the transport-level settings of opts into [request.Params].
// It carries the HTTP client, base URL, user agent, token, and headers;
// provider, model, and other task-level settings have no transport
// representation and are not projected. The request context is passed
// explicitly to the request functions and is not carried by Params.
func Params(opts hfopts.Options) request.Params {
	return request.Params{
		HTTPClient: opts.HTTPClient,
		BaseURL:    opts.BaseURL,
		UserAgent:  opts.UserAgent,
		Token:      opts.Token,
		Headers:    opts.Headers,
	}
}
