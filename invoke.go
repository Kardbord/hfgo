package hfgo

import (
	"encoding/json"
	"net/http"

	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/providers"
)

const errProviderMustNotBeNil = "provider must not be nil"

// modelDispatchConfig holds the common validation and dispatch parameters
// for doJSONInference and doStreamingInference.
type modelDispatchConfig struct {
	opts     request.Options
	task     providers.Task
	endpoint string
}

// resolveModelDispatch validates options and resolves the endpoint.
func resolveModelDispatch(opts request.Options, task providers.Task) (modelDispatchConfig, error) {
	var cfg modelDispatchConfig

	if opts.Model == "" {
		return cfg, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: "the model option must be set for " + string(task) + " to succeed",
			Err:     nil,
		}
	}

	if opts.Provider == nil {
		return cfg, &SDKError{
			Kind:    SDKErrorKindConfiguration,
			Message: errProviderMustNotBeNil,
			Err:     nil,
		}
	}

	endpoint, err := opts.Provider.Endpoint(task, opts.Model)
	if err != nil {
		return cfg, err
	}

	cfg.opts = opts
	cfg.task = task
	cfg.endpoint = endpoint

	return cfg, nil
}

// encodeRequest marshals a typed request and applies the provider wire-format
// transform. It returns the transformed body and the content-type to use when
// sending.
func encodeRequest[Req any](
	cfg *modelDispatchConfig, req Req,
) (body []byte, contentType string, err error) {
	hfBody, err := json.Marshal(req)
	if err != nil {
		return nil, "", &SDKError{
			Kind:    SDKErrorKindSerialization,
			Message: "failed to marshal request body",
			Err:     err,
		}
	}

	providerBody, ct, err := cfg.opts.Provider.EncodeRequest(cfg.task, hfBody, "application/json")
	if err != nil {
		return nil, "", err
	}

	return providerBody, ct, nil
}

// doJSONInference posts req to the inference model endpoint for the model
// configured in opts, returning the typed response. It returns a configuration
// SDK error when no model is set. task names the inference task in that error
// message, e.g. "text-classification" or "summarization".
//
// It applies the provider's wire-format transform to the request before
// sending and to the response after receiving.
func doJSONInference[Req, Resp any](
	opts request.Options,
	task providers.Task,
	req Req,
) (Resp, error) {
	var zero Resp

	cfg, err := resolveModelDispatch(opts, task)
	if err != nil {
		return zero, err
	}

	providerBody, ct, err := encodeRequest(&cfg, req)
	if err != nil {
		return zero, err
	}

	cfg.opts = cfg.opts.WithDefaultHeader("Content-Type", ct)
	cfg.opts = cfg.opts.WithDefaultHeader("Accept", "application/json")

	//nolint:bodyclose // DrainAndCloseBody closes the body.
	httpResp, err := request.DoBytes(cfg.opts, http.MethodPost, cfg.endpoint, providerBody)
	if err != nil {
		return zero, err
	}
	defer request.DrainAndCloseBody(httpResp.Body)

	if err := request.ValidateJSONResponseContentType(httpResp.Header); err != nil {
		return zero, err
	}

	respBody, err := request.ReadResponseBody(httpResp, cfg.opts.MaxResponseBodyBytes)
	if err != nil {
		return zero, err
	}

	respCT := httpResp.Header.Get("Content-Type")
	hfRespBody, _, err := cfg.opts.Provider.DecodeResponse(cfg.task, respBody, respCT)
	if err != nil {
		return zero, err
	}

	if err := json.Unmarshal(hfRespBody, &zero); err != nil {
		return zero, &SDKError{
			Kind:    SDKErrorKindSerialization,
			Message: "failed to decode response body",
			Err:     err,
		}
	}

	return zero, nil
}

// doRawInference sends an arbitrary request body and returns the raw response,
// applying provider wire-format transforms. It is the entry point for tasks
// with non-JSON request or response bodies (e.g. text-to-image, image-classification).
//
//nolint:unused // Entry point for future binary task services.
func doRawInference(
	opts request.Options,
	task providers.Task,
	body []byte,
	contentType string,
	accept string, // expected response Content-Type (e.g. "image/png")
) ([]byte, string, error) {
	cfg, err := resolveModelDispatch(opts, task)
	if err != nil {
		return nil, "", err
	}

	providerBody, ct, err := cfg.opts.Provider.EncodeRequest(cfg.task, body, contentType)
	if err != nil {
		return nil, "", err
	}

	cfg.opts = cfg.opts.WithDefaultHeader("Content-Type", ct)
	cfg.opts = cfg.opts.WithDefaultHeader("Accept", accept)

	//nolint:bodyclose // DrainAndCloseBody closes the body.
	httpResp, err := request.DoBytes(cfg.opts, http.MethodPost, cfg.endpoint, providerBody)
	if err != nil {
		return nil, "", err
	}
	defer request.DrainAndCloseBody(httpResp.Body)

	respBody, err := request.ReadResponseBody(httpResp, cfg.opts.MaxResponseBodyBytes)
	if err != nil {
		return nil, "", err
	}

	respCT := httpResp.Header.Get("Content-Type")
	hfBody, hfCT, err := cfg.opts.Provider.DecodeResponse(cfg.task, respBody, respCT)
	if err != nil {
		return nil, "", err
	}

	return hfBody, hfCT, nil
}

// doStreamingInference sends a typed request and returns a JSON stream
// of decoded SSE events, applying provider wire-format transforms.
func doStreamingInference[Req, T any](
	opts request.Options,
	task providers.Task,
	req Req,
) (*request.JSONStream[T], error) {
	cfg, err := resolveModelDispatch(opts, task)
	if err != nil {
		return nil, err
	}

	providerBody, ct, err := encodeRequest(&cfg, req)
	if err != nil {
		return nil, err
	}

	cfg.opts = cfg.opts.WithDefaultHeader("Content-Type", ct)
	cfg.opts = cfg.opts.WithDefaultHeader("Accept", "text/event-stream")

	//nolint:bodyclose // Body ownership transferred to RawStream.
	httpResp, err := request.DoBytes(cfg.opts, http.MethodPost, cfg.endpoint, providerBody)
	if err != nil {
		return nil, err
	}

	if err := request.ValidateEventStreamResponseContentType(httpResp.Header); err != nil {
		request.DrainAndCloseBody(httpResp.Body)

		return nil, err
	}

	raw, err := request.StreamRaw(cfg.opts.Context(), httpResp.Body)
	if err != nil {
		return nil, err
	}

	return request.NewJSONStream[T](raw, func(data []byte) ([]byte, error) {
		hfData, _, decodeErr := cfg.opts.Provider.DecodeResponse(cfg.task, data, "application/json")

		return hfData, decodeErr
	}), nil
}
