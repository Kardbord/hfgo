//go:build !integration

package hferrors

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEndOfStreamError(t *testing.T) {
	t.Parallel()

	require.NotEmpty(t, EndOfStreamError{}.Error())

	// Value, pointer, and wrapped forms all match via errors.Is.
	sameValue := EndOfStreamError{}
	require.ErrorIs(t, sameValue, EndOfStreamError{})
	require.ErrorIs(t, &EndOfStreamError{}, EndOfStreamError{})

	wrapped := fmt.Errorf("message_stop: %w", EndOfStreamError{})
	require.ErrorIs(t, wrapped, EndOfStreamError{})

	// The other signal and plain errors must not match.
	require.NotErrorIs(t, SkipEventError{}, EndOfStreamError{})
	require.NotErrorIs(t, errors.New("boom"), EndOfStreamError{})
}

func TestSkipEventError(t *testing.T) {
	t.Parallel()

	require.NotEmpty(t, SkipEventError{}.Error())

	sameValue := SkipEventError{}
	require.ErrorIs(t, sameValue, SkipEventError{})
	require.ErrorIs(t, &SkipEventError{}, SkipEventError{})

	wrapped := fmt.Errorf("content_block_start: %w", SkipEventError{})
	require.ErrorIs(t, wrapped, SkipEventError{})

	require.NotErrorIs(t, EndOfStreamError{}, SkipEventError{})
	require.NotErrorIs(t, errors.New("boom"), SkipEventError{})
}
