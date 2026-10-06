package task

import (
	"context"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/internal/request"
)

// eventContentType is the Content-Type synthesized for SSE event payloads:
// event frames carry no media type of their own, so each payload is assumed
// to be JSON.
const eventContentType = "application/json"

// validateDispatch validates options for a typed task dispatch: the full
// Options.Validate checks plus model presence. Endpoint methods are
// documented to receive a non-empty Model as a result.
func validateDispatch(opts hfopts.Options) error {
	if err := opts.Validate(); err != nil {
		return err
	}

	if opts.Model == "" {
		return &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindConfiguration,
			Message: "model must not be empty",
			Err:     nil,
		}
	}

	return nil
}

// applyCodecHeaders applies codec-provided request headers as defaults:
// headers the caller already set on opts win. Authorization is dropped:
// authentication is caller-owned (Options.Token / Options.Headers), so a
// codec-supplied value must never replace the bearer token.
func applyCodecHeaders(opts hfopts.Options, headers http.Header) hfopts.Options {
	for key, values := range headers {
		canonical := http.CanonicalHeaderKey(key)
		if canonical == "Authorization" || len(opts.Headers.Values(key)) > 0 {
			continue
		}

		opts = opts.With(hfopts.WithHeaders(http.Header{
			canonical: append([]string(nil), values...),
		}))
	}

	return opts
}

// doInference posts req to the inference model endpoint using the given codec,
// returning the typed response.
func doInference[Req, Resp any](
	opts hfopts.Options,
	endpoint string,
	codec hfproviders.Codec[Req, Resp],
	req Req,
) (Resp, error) {
	var zero Resp

	providerBody, providerHeaders, err := codec.Encode(hfproviders.EncodeParams[Req]{
		Context: opts.Context(),
		Request: req,
		Model:   opts.Model,
	})
	if err != nil {
		return zero, err
	}

	opts = applyCodecHeaders(opts, providerHeaders)

	//nolint:bodyclose // DrainAndCloseBody closes the body.
	httpResp, err := request.DoBytes(opts, http.MethodPost, endpoint, providerBody)
	if err != nil {
		return zero, err
	}
	defer request.DrainAndCloseBody(httpResp.Body)

	respBody, err := request.DecodeHTTPResponse(httpResp, opts.MaxResponseBodyBytes)
	if err != nil {
		return zero, err
	}
	if respBody == nil {
		return zero, nil // 204/205
	}

	resp, err := codec.Decode(hfproviders.DecodeParams{
		Context: opts.Context(),
		Body:    respBody,
		Headers: httpResp.Header,
	})
	if err != nil {
		return zero, err
	}

	return resp, nil
}

// doStreamingInference sends a typed request and returns a stream of decoded
// events using the given codec. Accept is forced to text/event-stream because
// SSE framing is a transport concern, not a codec concern; codec-supplied
// headers apply as defaults for everything else.
func doStreamingInference[Req, T any](
	opts hfopts.Options,
	endpoint string,
	codec hfproviders.Codec[Req, T],
	req Req,
) (*request.JSONStream[T], error) {
	providerBody, providerHeaders, err := codec.Encode(hfproviders.EncodeParams[Req]{
		Context: opts.Context(),
		Request: req,
		Model:   opts.Model,
	})
	if err != nil {
		return nil, err
	}

	opts = applyCodecHeaders(opts, providerHeaders)
	opts = opts.With(hfopts.WithHeader("Accept", "text/event-stream"))

	//nolint:bodyclose // Body ownership transferred to RawStream.
	httpResp, err := request.DoBytes(opts, http.MethodPost, endpoint, providerBody)
	if err != nil {
		return nil, err
	}

	if err := request.ValidateEventStreamResponseContentType(httpResp.Header); err != nil {
		request.DrainAndCloseBody(httpResp.Body)

		return nil, err
	}

	raw, err := request.StreamRaw(opts.Context(), httpResp.Body)
	if err != nil {
		request.DrainAndCloseBody(httpResp.Body)

		return nil, err
	}

	return request.NewJSONStream[T](raw, func(ctx context.Context, data []byte) (T, error) {
		return codec.Decode(hfproviders.DecodeParams{
			Context: ctx,
			Body:    data,
			Headers: http.Header{"Content-Type": {eventContentType}},
		})
	}), nil
}
