//go:build !integration

package hfproviders

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/internal/request"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// TestQueueRetryAfter exercises the shared Retry-After parser as a table:
// seconds are honored, garbage and non-positive values are ignored, and the
// cap bounds pathological values.
func TestQueueRetryAfter(t *testing.T) {
	t.Parallel()

	cases := []struct {
		header string
		want   time.Duration
	}{
		{header: "3", want: 3 * time.Second},
		{header: " 3 ", want: 3 * time.Second},
		{header: "soon", want: 0},
		{header: "0", want: 0},
		{header: "-3", want: 0},
		{header: "999999", want: queueMaxRetryAfter},
		{header: "", want: 0},
	}

	for _, tc := range cases {
		headers := http.Header{}
		if tc.header != "" {
			headers.Set("Retry-After", tc.header)
		}

		require.Equal(t, tc.want, queueRetryAfter(headers), "Retry-After %q", tc.header)
	}
}

// TestQueueTransientStatus exercises the shared transient-status predicate:
// rate limiting and server errors are transient, everything else is not.
func TestQueueTransientStatus(t *testing.T) {
	t.Parallel()

	require.True(t, queueTransientStatus(http.StatusTooManyRequests))
	require.True(t, queueTransientStatus(http.StatusInternalServerError))
	require.True(t, queueTransientStatus(http.StatusBadGateway))
	require.False(t, queueTransientStatus(http.StatusOK))
	require.False(t, queueTransientStatus(http.StatusBadRequest))
	require.False(t, queueTransientStatus(http.StatusNotFound))
}

// TestQueueIsContextError exercises the shared context-error predicate:
// context cancellation and deadline expiry count, including when wrapped;
// other errors do not.
func TestQueueIsContextError(t *testing.T) {
	t.Parallel()

	require.True(t, queueIsContextError(context.Canceled))
	require.True(t, queueIsContextError(context.DeadlineExceeded))
	require.True(t, queueIsContextError(fmt.Errorf("wrapped: %w", context.Canceled)))
	require.False(t, queueIsContextError(errors.New("connection reset")))
}

// TestQueuePollOnce_TransientNoOverride exercises the shared poll exchange's
// transport predicate wiring: a transient poll failure yields no task, no
// error, and a zero wait override when the server sends no Retry-After.
func TestQueuePollOnce_TransientNoOverride(t *testing.T) {
	t.Parallel()

	// A single 503 response with no Retry-After suffices: one poll attempt,
	// no follow-up request.
	mt := testutils.NewMockTransport(http.StatusServiceUnavailable, "", nil)
	client := testutils.NewMockHTTPClient(mt)

	pollParams := request.Params{
		HTTPClient: &client,
		BaseURL:    "https://queue.example.com",
		UserAgent:  "test-agent/1.0",
		Headers:    http.Header{},
	}

	out, override, err := queuePollOnce(
		context.Background(), PostSubmitParams{}, pollParams, "/task",
		queueExchange[string]{
			decode:   func(b []byte) (string, error) { return string(b), nil },
			terminal: func(*string) (*string, bool, error) { return nil, true, nil },
		},
	)
	require.NoError(t, err)
	require.Nil(t, out)
	require.Equal(t, time.Duration(0), override)
}

// TestQueueDecodeMediaBody exercises the shared output-media validation: a
// matching advertised type is kept verbatim, a generic or mismatched type is
// byte-sniffed and accepted when the sniff agrees with the prefix, and the
// advertised type is rejected only when the bytes agree with it or cannot be
// identified at all.
func TestQueueDecodeMediaBody(t *testing.T) {
	t.Parallel()

	// pngBody is the full PNG file signature, which http.DetectContentType
	// reports as image/png.
	pngBody := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	t.Run("matching advertised type is kept", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"image/png"}},
		}

		body, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, pngBody, body)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("generic advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"application/octet-stream"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("mismatched advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"text/plain"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("unparseable advertised type is rescued by sniffing", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    pngBody,
			Headers: http.Header{"Content-Type": {"%%%"}},
		}

		_, mediaType, err := queueDecodeMediaBody(params, "image/")
		require.NoError(t, err)
		require.Equal(t, "image/png", mediaType)
	})

	t.Run("body contradicting the advertised type is rejected", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    []byte(`{"foo":"bar"}`),
			Headers: http.Header{"Content-Type": {"application/octet-stream"}},
		}

		_, _, err := queueDecodeMediaBody(params, "image/")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "image/")
		require.ErrorContains(t, err, "application/octet-stream")
	})

	t.Run("unidentifiable body is rejected", func(t *testing.T) {
		t.Parallel()

		params := DecodeParams{
			Context: context.Background(),
			Body:    []byte("not an image"),
			Headers: http.Header{"Content-Type": {"text/plain"}},
		}

		_, _, err := queueDecodeMediaBody(params, "image/")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "text/plain")
	})
}

// TestQueueTiming exercises the schedule-merge predicate: a non-positive
// override keeps the default, a positive override wins per bound.
func TestQueueTiming(t *testing.T) {
	t.Parallel()

	defaults := queueSchedule{
		initial: 2 * time.Second,
		step:    1 * time.Second,
		max:     5 * time.Second,
	}

	got := queueTiming(defaults, queueSchedule{})
	require.Equal(t, defaults, got)

	overrides := queueSchedule{
		initial: 10 * time.Millisecond,
		step:    20 * time.Millisecond,
		max:     30 * time.Millisecond,
	}
	require.Equal(t, overrides, queueTiming(defaults, overrides))

	// Zero overrides fall back to the defaults individually.
	require.Equal(t, defaults, queueTiming(defaults, queueSchedule{initial: 0}))
	require.Equal(t, defaults, queueTiming(defaults, queueSchedule{step: 0}))
	require.Equal(t, defaults, queueTiming(defaults, queueSchedule{max: 0}))
}

// TestQueueTaskCodecTiming exercises the codec's schedule knob wiring: a
// codec carrying only defaults selects them, and each knob overrides only its
// own bound.
func TestQueueTaskCodecTiming(t *testing.T) {
	t.Parallel()

	t.Run("defaults only", func(t *testing.T) {
		t.Parallel()

		codec := queueTaskCodec{defaults: falaiDefaultPollTiming()}
		require.Equal(t, falaiDefaultPollTiming(), codec.timing())
	})

	t.Run("knobs override individually", func(t *testing.T) {
		t.Parallel()

		codec := queueTaskCodec{
			defaults:     falaiDefaultPollTiming(),
			pollInterval: 10 * time.Millisecond,
		}
		timing := codec.timing()
		require.Equal(t, 10*time.Millisecond, timing.initial)
		require.Equal(t, falaiPollIntervalStep, timing.step)
		require.Equal(t, falaiMaxPollInterval, timing.max)

		codec = queueTaskCodec{
			defaults: falaiDefaultPollTiming(),
			pollStep: 20 * time.Millisecond,
		}
		timing = codec.timing()
		require.Equal(t, falaiInitialPollInterval, timing.initial)
		require.Equal(t, 20*time.Millisecond, timing.step)
		require.Equal(t, falaiMaxPollInterval, timing.max)

		codec = queueTaskCodec{
			defaults:        falaiDefaultPollTiming(),
			maxPollInterval: 30 * time.Millisecond,
		}
		timing = codec.timing()
		require.Equal(t, falaiInitialPollInterval, timing.initial)
		require.Equal(t, falaiPollIntervalStep, timing.step)
		require.Equal(t, 30*time.Millisecond, timing.max)
	})
}

// TestQueuePollBase exercises the shared follow-up base resolution: the
// router gains the prefix, case is ignored, already-routed task URLs skip
// the prefix, and unparseable bases are configuration errors.
func TestQueuePollBase(t *testing.T) {
	t.Parallel()

	t.Run("router host gains prefix", func(t *testing.T) {
		t.Parallel()

		base, routed, err := queuePollBase("https://router.huggingface.co", "/fal-ai", "")
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://router.huggingface.co/fal-ai", base)
	})

	t.Run("router base with trailing slash", func(t *testing.T) {
		t.Parallel()

		base, routed, err := queuePollBase("https://router.huggingface.co/", "/fal-ai", "")
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://router.huggingface.co/fal-ai", base)
	})

	t.Run("router host matched case-insensitively", func(t *testing.T) {
		t.Parallel()

		base, routed, err := queuePollBase("https://Router.HuggingFace.CO", "/fal-ai", "")
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://Router.HuggingFace.CO/fal-ai", base)
	})

	t.Run("other hosts resolve as-is", func(t *testing.T) {
		t.Parallel()

		base, routed, err := queuePollBase("https://queue.fal.run", "/fal-ai", "")
		require.NoError(t, err)
		require.False(t, routed)
		require.Equal(t, "https://queue.fal.run", base)
	})

	t.Run("already-routed task URL skips prefix", func(t *testing.T) {
		t.Parallel()

		base, routed, err := queuePollBase(
			"https://router.huggingface.co", "/replicate", "router.huggingface.co",
		)
		require.NoError(t, err)
		require.True(t, routed)
		require.Equal(t, "https://router.huggingface.co", base)
	})

	t.Run("invalid base URL is an error", func(t *testing.T) {
		t.Parallel()

		_, _, err := queuePollBase("http://[::1:bad", "/fal-ai", "")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
	})
}

// TestQueueParseTaskURL exercises the shared task-URL extraction: absolute
// URLs keep only path and query, relative paths gain a leading slash, an
// empty path normalizes to the root, and unparseable URLs are serialization
// errors.
func TestQueueParseTaskURL(t *testing.T) {
	t.Parallel()

	t.Run("absolute URL resolves from its path", func(t *testing.T) {
		t.Parallel()

		task, err := queueParseTaskURL(
			"https://queue.fal.run/fal-ai/requests/req_1", "fal.ai response URL",
		)
		require.NoError(t, err)
		require.Equal(t, "/fal-ai/requests/req_1", task.path)
		require.Empty(t, task.query)
		require.Equal(t, "queue.fal.run", task.host)
	})

	t.Run("query preserved", func(t *testing.T) {
		t.Parallel()

		task, err := queueParseTaskURL(
			"https://queue.fal.run/fal-ai/requests/req_1?sig=abc&exp=2", "fal.ai response URL",
		)
		require.NoError(t, err)
		require.Equal(t, "/fal-ai/requests/req_1", task.path)
		require.Equal(t, "sig=abc&exp=2", task.query)
	})

	t.Run("relative path gains a leading slash", func(t *testing.T) {
		t.Parallel()

		task, err := queueParseTaskURL("fal-ai/requests/req_1", "fal.ai response URL")
		require.NoError(t, err)
		require.Equal(t, "/fal-ai/requests/req_1", task.path)
		require.Empty(t, task.query)
		require.Empty(t, task.host)
	})

	t.Run("empty path normalizes to root", func(t *testing.T) {
		t.Parallel()

		task, err := queueParseTaskURL("https://queue.fal.run", "fal.ai response URL")
		require.NoError(t, err)
		require.Equal(t, "/", task.path)
	})

	t.Run("unparseable URL is an error", func(t *testing.T) {
		t.Parallel()

		_, err := queueParseTaskURL("http://h/%zz", "fal.ai response URL")
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindSerialization)
		require.ErrorContains(t, err, "invalid fal.ai response URL")
	})
}

// TestQueuePollTarget exercises the shared follow-up request assembly: the
// caller's client, credentials, User-Agent, and JSON Accept header make it
// through, and the task URL query rejoins the path.
func TestQueuePollTarget(t *testing.T) {
	t.Parallel()

	client := &http.Client{}
	params := PostSubmitParams{
		HTTPClient: client,
		BaseURL:    "https://router.huggingface.co",
		Token:      "tok",
		UserAgent:  "agent/1.0",
	}

	pollParams, path, err := queuePollTarget(
		params, "/fal-ai/requests/req_1?sig=abc", "/fal-ai", "fal.ai response URL",
	)
	require.NoError(t, err)
	require.Same(t, client, pollParams.HTTPClient)
	require.Equal(t, "https://router.huggingface.co/fal-ai", pollParams.BaseURL)
	require.Equal(t, "tok", pollParams.Token)
	require.Equal(t, "agent/1.0", pollParams.UserAgent)
	gotAccept := pollParams.Headers.Get(headerAccept)
	require.Equal(t, ctJSON, gotAccept) //nolint:testifylint // media type
	require.Equal(t, "/fal-ai/requests/req_1?sig=abc", path)
}

// TestQueuePollWait exercises the shared poll wait: an already-cancelled
// context returns its error immediately, and a satisfied wait returns nil.
func TestQueuePollWait(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, queuePollWait(ctx, time.Hour), context.Canceled)
	require.NoError(t, queuePollWait(context.Background(), time.Millisecond))
}

// TestQueueAwait exercises the shared polling loop's terminal classification:
// a terminal first round returns immediately, a transient round keeps
// polling, and a context expiry surfaces rather than polling further.
func TestQueueAwait(t *testing.T) {
	t.Parallel()

	t.Run("terminates on the first terminal round", func(t *testing.T) {
		t.Parallel()

		got, err := queueAwait(context.Background(), queueSchedule{},
			func(context.Context) (*string, time.Duration, error) {
				out := "done"

				return &out, 0, nil
			})
		require.NoError(t, err)
		require.Equal(t, "done", *got)
	})

	t.Run("keeps polling through transient rounds", func(t *testing.T) {
		t.Parallel()

		var calls atomic.Int64
		got, err := queueAwait(context.Background(), queueSchedule{initial: time.Millisecond},
			func(context.Context) (*string, time.Duration, error) {
				if calls.Add(1) < 3 {
					return nil, 0, nil
				}
				out := "done"

				return &out, 0, nil
			})
		require.NoError(t, err)
		require.Equal(t, "done", *got)
		require.Equal(t, int64(3), calls.Load())
	})

	t.Run("context expiry stops polling", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		pollOnce := func(context.Context) (*string, time.Duration, error) {
			t.Fatal("must not poll after context expiry")

			return nil, 0, nil
		}
		_, err := queueAwait(ctx, queueSchedule{initial: time.Hour}, pollOnce)
		require.ErrorIs(t, err, context.Canceled)
	})
}

// TestQueueFetchRetry exercises the shared result-fetch retry loop: the first
// attempt fires immediately, transient failures retry, and a non-transient
// body is returned as-is.
func TestQueueFetchRetry(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	body, err := queueFetchRetry(context.Background(), queueSchedule{initial: time.Millisecond},
		func(context.Context) ([]byte, bool, time.Duration, error) {
			if calls.Add(1) < 3 {
				return nil, true, 0, nil
			}

			return []byte("result"), false, 0, nil
		})
	require.NoError(t, err)
	require.Equal(t, []byte("result"), body)
	require.Equal(t, int64(3), calls.Load())
}

// TestQueueDownloadOutput exercises the shared unauthenticated output
// download: the request carries no Authorization header, and a missing
// Content-Type is recovered by byte sniffing.
func TestQueueDownloadOutput(t *testing.T) {
	t.Parallel()

	png := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write(png)
	}))
	defer server.Close()

	body, headers, err := queueDownloadOutput(context.Background(),
		PostSubmitParams{HTTPClient: server.Client(), MaxResponseBodyBytes: 1 << 20},
		server.URL+"/output.png",
	)
	require.NoError(t, err)
	require.Equal(t, png, body)
	require.Empty(t, gotAuth, "output downloads must not carry the caller's token")
	require.Equal(t, "image/png", headers.Get("Content-Type"))
}

// TestQueueDownloadOutput_MalformedContentType verifies that a response whose
// advertised Content-Type cannot be parsed falls back to byte sniffing.
func TestQueueDownloadOutput_MalformedContentType(t *testing.T) {
	t.Parallel()

	png := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "not a media type")
		_, _ = w.Write(png)
	}))
	defer server.Close()

	body, headers, err := queueDownloadOutput(context.Background(),
		PostSubmitParams{HTTPClient: server.Client(), MaxResponseBodyBytes: 1 << 20},
		server.URL+"/output.png",
	)
	require.NoError(t, err)
	require.Equal(t, png, body)
	require.Equal(t, "image/png", headers.Get("Content-Type"))
}

// TestQueueErrorHelpers exercises the shared task-error constructors: a
// failed task surfaces as a provider-kind error, and a decode failure as a
// serialization-kind error carrying the provider label.
func TestQueueErrorHelpers(t *testing.T) {
	t.Parallel()

	err := queueTaskFailedError("WaveSpeed task failed: boom")
	testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindProvider)
	require.ErrorContains(t, err, "WaveSpeed task failed: boom")

	decodeErr := queueDecodeTaskError("WaveSpeed task response", errors.New("not json"))
	testutils.AssertSDKErrorKind(t, decodeErr, hferrors.SDKErrorKindSerialization)
	require.ErrorContains(t, decodeErr, "failed to decode WaveSpeed task response")
}
