//go:build !integration

package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

type sample struct{}

func TestIsNil(t *testing.T) {
	t.Parallel()

	var nilAny any
	var nilPtr *sample
	var nilMap map[string]string
	var nilSlice []string
	var nilChan chan int
	var nilFunc func()
	var nilIface error

	require.True(t, IsNil(nilAny))
	require.True(t, IsNil(nilPtr))
	require.True(t, IsNil(nilMap))
	require.True(t, IsNil(nilSlice))
	require.True(t, IsNil(nilChan))
	require.True(t, IsNil(nilFunc))
	require.True(t, IsNil(nilIface))

	// Typed nils wrapped in interfaces are still nil.
	var asAny any = nilPtr
	var asError error = (*sampleError)(nil)
	require.True(t, IsNil(asAny))
	require.True(t, IsNil(asError))

	require.False(t, IsNil(sample{}))
	require.False(t, IsNil(&sample{}))
	require.False(t, IsNil(""))
}

type sampleError struct{}

func (*sampleError) Error() string { return "boom" }
