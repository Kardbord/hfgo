//go:build !integration

package transport_test

import (
	"net/http"
	"testing"

	"github.com/Kardbord/hfgo/v4/hfopts"
	"github.com/Kardbord/hfgo/v4/internal/transport"
	"github.com/stretchr/testify/require"
)

func TestParamsProjectsTransportFields(t *testing.T) {
	t.Parallel()

	client := &http.Client{}
	opts := hfopts.NewOptions().With(
		hfopts.WithBaseURL("https://hub.example"),
		hfopts.WithToken("token"),
		hfopts.WithUserAgent("test/1.0"),
		hfopts.WithHeaders(http.Header{"X-Test": {"one"}}),
		hfopts.WithHTTPClientFactory(func() http.Client { return *client }),
	)

	params := transport.Params(opts)

	require.Same(t, opts.HTTPClient, params.HTTPClient)
	require.Equal(t, opts.BaseURL, params.BaseURL)
	require.Equal(t, opts.UserAgent, params.UserAgent)
	require.Equal(t, opts.Token, params.Token)
	require.Equal(t, opts.Headers, params.Headers)
}
