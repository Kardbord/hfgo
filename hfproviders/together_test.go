//go:build !integration

package hfproviders

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
	"github.com/Kardbord/hfgo/v4/internal/testutils"
	"github.com/stretchr/testify/require"
)

// togetherChatRequest builds the minimal valid chat request; the dispatch
// layer always fills Model before the codec sees it.
func togetherChatRequest() hftypes.ChatRequest {
	text := "hi"

	return hftypes.ChatRequest{
		Model: new("deepseek-ai/DeepSeek-V4.1-Flash:together"),
		Messages: []hftypes.ChatMessage{
			{
				Role:    "user",
				Content: hftypes.ChatMessageContent{Text: &text},
			},
		},
	}
}

func TestTogetherProvider_NameAndSuffix(t *testing.T) {
	t.Parallel()

	provider := NewTogetherProvider()

	require.Equal(t, "together", provider.Name())
	require.Equal(t, "together", provider.ProviderSuffix())
}

// TestTogetherProvider_Contract pins the task interfaces TogetherProvider
// serves and the ones it must reject.
func TestTogetherProvider_Contract(t *testing.T) {
	t.Parallel()

	provider := NewTogetherProvider()

	requireServedTaskInterfaces(t, provider, []func(Provider) error{
		func(p Provider) error {
			_, err := AsProvider[ChatProvider](p)

			return err
		},
		func(p Provider) error {
			_, err := AsProvider[ChatStreamProvider](p)

			return err
		},
	})

	requireMissingTaskInterfaces(t, provider, togetherProviderName, []taskCast{
		{
			typeName: "hfproviders.SpeechRecognitionProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SpeechRecognitionProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.SummarizationProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[SummarizationProvider](p)

				return err
			},
		},
		{
			typeName: "hfproviders.TextToImageProvider",
			cast: func(p Provider) error {
				_, err := AsProvider[TextToImageProvider](p)

				return err
			},
		},
	})
}

func TestTogetherProvider_ChatEndpoint(t *testing.T) {
	t.Parallel()

	provider := NewTogetherProvider()

	// The shared chat half of the partner contract: the model-independent
	// OpenAI-compatible endpoint, the built-in JSON codecs, and the
	// ProviderParams merge.
	requireOpenAIChatPartnerEndpoint(t, provider, togetherProviderName)
}

// TestTogetherProvider_ResponseFormatPassthrough pins that chat requests
// encode response formats exactly as [hftypes.ChatRequest] marshals them,
// including OpenAI's nested json_schema envelope that Together's
// chat-completions API consumes.
func TestTogetherProvider_ResponseFormatPassthrough(t *testing.T) {
	t.Parallel()

	provider := NewTogetherProvider()

	encode := func(t *testing.T, req hftypes.ChatRequest) map[string]json.RawMessage {
		t.Helper()

		chat, err := AsProvider[ChatProvider](provider)
		require.NoError(t, err)

		body, _, err := chat.ChatCodec().
			Encode(context.Background(), EncodeParams[hftypes.ChatRequest]{Request: req})
		require.NoError(t, err)

		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(body, &fields))

		return fields
	}

	t.Run("json_schema keeps the nested envelope", func(t *testing.T) {
		t.Parallel()

		schema := json.RawMessage(
			`{"type":"object","properties":{"answer":{"type":"string"}}}`,
		)
		req := togetherChatRequest()
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
			JSONSchema: &hftypes.ChatJSONSchemaConfig{
				Name:   "answer",
				Strict: new(true),
				Schema: schema,
			},
		}

		fields := encode(t, req)

		// The envelope keeps name, strict, and the schema verbatim.
		const envelopePrefix = `{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":`

		require.JSONEq(t,
			envelopePrefix+string(schema)+`}}`,
			string(fields["response_format"]),
		)
	})

	t.Run("text and json_object pass through", func(t *testing.T) {
		t.Parallel()

		for _, formatType := range []hftypes.ResponseFormatType{
			hftypes.ResponseFormatTypeText,
			hftypes.ResponseFormatTypeJSONObject,
		} {
			req := togetherChatRequest()
			req.ResponseFormat = &hftypes.ChatResponseFormat{Type: formatType}

			fields := encode(t, req)
			require.JSONEq(t,
				`{"type":"`+string(formatType)+`"}`,
				string(fields["response_format"]),
			)
		}
	})

	t.Run("provider-specific format types pass through", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatType("audio"),
		}

		fields := encode(t, req)
		require.JSONEq(t, `{"type":"audio"}`, string(fields["response_format"]))
	})

	t.Run("absent response format yields no key", func(t *testing.T) {
		t.Parallel()

		fields := encode(t, togetherChatRequest())
		_, present := fields["response_format"]
		require.False(t, present)
	})

	t.Run("invalid request fails during encoding", func(t *testing.T) {
		t.Parallel()

		req := togetherChatRequest()
		// json_schema requires the envelope, so this fails MarshalJSON.
		req.ResponseFormat = &hftypes.ChatResponseFormat{
			Type: hftypes.ResponseFormatTypeJSONSchema,
		}

		chat, err := AsProvider[ChatProvider](provider)
		require.NoError(t, err)

		_, _, err = chat.ChatCodec().Encode(context.Background(), EncodeParams[hftypes.ChatRequest]{
			Request: req,
		})
		testutils.AssertSDKErrorKind(t, err, hferrors.SDKErrorKindConfiguration)
		require.ErrorContains(t, err, "json_schema requires json_schema field")
	})
}
