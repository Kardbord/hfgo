package task

import (
	"context"
	"net/http"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/hfproviders"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/transport"
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
		if canonical == "Authorization" || len(opts.Headers.Values(canonical)) > 0 {
			continue
		}

		opts = opts.With(hfopts.WithHeaders(http.Header{
			canonical: append([]string(nil), values...),
		}))
	}

	return opts
}

// endpointMethod returns endpoint's HTTP method, defaulting to POST when the
// provider leaves it unset.
func endpointMethod(endpoint hfproviders.Endpoint) string {
	if endpoint.Method == "" {
		return http.MethodPost
	}

	return endpoint.Method
}

// doInference posts req to the inference model endpoint using the given codec,
// returning the typed response. The context carries the caller's deadline and
// cancellation signals for the whole exchange, including any async PostSubmit
// follow-up requests.
func doInference[Req, Resp any](
	ctx context.Context,
	opts hfopts.Options,
	endpoint hfproviders.Endpoint,
	codec hfproviders.Codec[Req, Resp],
	req Req,
) (Resp, error) {
	var zero Resp

	providerBody, providerHeaders, err := codec.Encode(ctx, hfproviders.EncodeParams[Req]{
		Request: req,
		Model:   opts.Model,
		Mapping: endpoint.Mapping,
	})
	if err != nil {
		return zero, err
	}

	opts = applyCodecHeaders(opts, providerHeaders)

	//nolint:bodyclose // DrainAndCloseBody closes the body.
	httpResp, err := request.DoBytes(
		ctx,
		transport.Params(opts),
		endpointMethod(endpoint),
		endpoint.Path,
		providerBody,
	)
	if err != nil {
		return zero, err
	}
	defer request.DrainAndCloseBody(httpResp.Body, opts.MaxResponseBodyBytes)

	respBody, err := request.DecodeHTTPResponse(httpResp, opts.MaxResponseBodyBytes)
	if err != nil {
		return zero, err
	}
	if respBody == nil {
		return zero, nil // 204/205
	}

	// Async codecs own multi-step exchanges (e.g. submit-poll-retrieve) after
	// the initial response; plain codecs decode the response body directly.
	var resp Resp
	if asyncCodec, ok := any(codec).(hfproviders.AsyncCodec[Req, Resp]); ok {
		resp, err = asyncCodec.PostSubmit(
			ctx,
			hfproviders.PostSubmitParams{
				HTTPClient:           opts.HTTPClient,
				BaseURL:              opts.BaseURL,
				Token:                opts.Token,
				UserAgent:            opts.UserAgent,
				MaxResponseBodyBytes: opts.MaxResponseBodyBytes,
			},
			respBody,
			httpResp.Header,
		)
	} else {
		resp, err = codec.Decode(ctx, hfproviders.DecodeParams{
			Body:    respBody,
			Event:   "", // unary responses have no SSE framing
			Headers: httpResp.Header,
		})
	}
	if err != nil {
		return zero, err
	}

	return resp, nil
}

// doStreamingInference sends a typed request and returns a stream of decoded
// events using the given codec. Accept is forced to text/event-stream because
// SSE framing is a transport concern, not a codec concern; codec-supplied
// headers apply as defaults for everything else. The context carries the
// caller's deadline and cancellation signals for the request and the returned
// stream's decoder.
func doStreamingInference[Req, T any](
	ctx context.Context,
	opts hfopts.Options,
	endpoint hfproviders.Endpoint,
	codec hfproviders.Codec[Req, T],
	req Req,
) (*request.JSONStream[T], error) {
	providerBody, providerHeaders, err := codec.Encode(ctx, hfproviders.EncodeParams[Req]{
		Request: req,
		Model:   opts.Model,
		Mapping: endpoint.Mapping,
	})
	if err != nil {
		return nil, err
	}

	opts = applyCodecHeaders(opts, providerHeaders)
	opts = opts.With(hfopts.WithHeader("Accept", "text/event-stream"))

	//nolint:bodyclose // Body ownership transferred to RawStream.
	httpResp, err := request.DoBytes(
		ctx,
		transport.Params(opts),
		endpointMethod(endpoint),
		endpoint.Path,
		providerBody,
	)
	if err != nil {
		return nil, err
	}

	if err := request.ValidateEventStreamResponseContentType(httpResp.Header); err != nil {
		request.DrainAndCloseBody(httpResp.Body, opts.MaxResponseBodyBytes)

		return nil, err
	}

	raw, err := request.StreamRawWithLimit(
		ctx,
		httpResp.Body,
		opts.MaxResponseBodyBytes,
	)
	if err != nil {
		// StreamRaw only rejects a nil body, which net/http does not surface
		// through Client.Do. Keep the defensive close for future changes.
		request.DrainAndCloseBody(httpResp.Body, opts.MaxResponseBodyBytes)

		return nil, err
	}

	// The decode hook intentionally uses the dispatch context rather than the
	// per-Recv context: Decode is pure CPU work, so no per-frame cancellation
	// is needed, and this matches the pre-refactor behavior where DecodeParams
	// received the dispatch-time context from Options.
	return request.NewJSONStream[T](raw, func(_ context.Context, ev request.RawEvent) (T, error) {
		return codec.Decode(ctx, hfproviders.DecodeParams{
			Body:    ev.Data,
			Event:   ev.Event,
			Headers: http.Header{"Content-Type": {eventContentType}},
		})
	}), nil
}
