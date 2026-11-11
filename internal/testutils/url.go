package testutils

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// AssertURL compares a raw URL string to the expected URL parts.
func AssertURL(t *testing.T, raw string, want *url.URL) {
	t.Helper()

	got, err := url.Parse(raw)
	require.NoError(t, err, "failed to parse URL %q", raw)

	assert.Equal(t, want.Scheme, got.Scheme)
	assert.Equal(t, want.Host, got.Host)
	assert.Equal(t, want.Path, got.Path)
	assert.Equal(t, want.RawQuery, got.RawQuery)
	assert.Equal(t, want.Fragment, got.Fragment)
}
