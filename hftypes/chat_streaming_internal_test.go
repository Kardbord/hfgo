//go:build !integration

package hftypes

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/stretchr/testify/require"
)

// mockChatStreamer is a chatstreamer implementation for testing ChatStream.Recv.
type mockChatStreamer struct {
	responses []ChatStreamResponse
	index     int
	closeErr  error
	closed    bool
}

func (m *mockChatStreamer) Recv(_ context.Context) (ChatStreamResponse, error) {
	if m.index >= len(m.responses) {
		return ChatStreamResponse{}, io.EOF
	}

	resp := m.responses[m.index]
	m.index++

	return resp, nil
}

func (m *mockChatStreamer) Close() error {
	m.closed = true

	return m.closeErr
}

func TestChatStream_Recv_MergesToolCallMetadata(t *testing.T) {
	t.Parallel()

	stream := NewChatStream(&mockChatStreamer{
		responses: []ChatStreamResponse{
			{
				Choices: []ChatStreamChoice{
					{
						Index: 0,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									ID:       "call_0",
									Type:     "function",
									Index:    0,
									Function: ChatStreamFunction{Name: "fn", Arguments: ""},
								},
							},
						},
					},
				},
			},
			{
				Choices: []ChatStreamChoice{
					{
						Index: 0,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									Index:    0,
									Function: ChatStreamFunction{Arguments: `{"foo":1}`},
								},
							},
						},
					},
				},
			},
		},
	})

	first, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)
	require.Equal(t, "function", first.Choices[0].Delta.ToolCalls[0].Type)
	require.Equal(t, "fn", first.Choices[0].Delta.ToolCalls[0].Function.Name)

	second, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", second.Choices[0].Delta.ToolCalls[0].ID)
	require.Equal(t, "function", second.Choices[0].Delta.ToolCalls[0].Type)
	require.Equal(t, "fn", second.Choices[0].Delta.ToolCalls[0].Function.Name)
	require.Equal(t, `{"foo":1}`, second.Choices[0].Delta.ToolCalls[0].Function.Arguments)
}

func TestChatStream_Recv_MergesAcrossChoices(t *testing.T) {
	t.Parallel()

	stream := NewChatStream(&mockChatStreamer{
		responses: []ChatStreamResponse{
			{
				Choices: []ChatStreamChoice{
					{
						Index: 0,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									ID:       "call_0",
									Type:     "function",
									Index:    0,
									Function: ChatStreamFunction{Name: "fn", Arguments: ""},
								},
							},
						},
					},
					{
						Index: 1,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									ID:       "call_1",
									Type:     "function",
									Index:    0,
									Function: ChatStreamFunction{Name: "fn2", Arguments: ""},
								},
							},
						},
					},
				},
			},
			{
				Choices: []ChatStreamChoice{
					{
						Index: 0,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									Index:    0,
									Function: ChatStreamFunction{Arguments: `{"foo":1}`},
								},
							},
						},
					},
					{
						Index: 1,
						Delta: ChatStreamDelta{
							ToolCalls: []ChatStreamToolCall{
								{
									Index:    0,
									Function: ChatStreamFunction{Arguments: `{"bar":2}`},
								},
							},
						},
					},
				},
			},
		},
	})

	first, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Len(t, first.Choices, 2)
	require.Equal(t, "call_0", first.Choices[0].Delta.ToolCalls[0].ID)
	require.Equal(t, "call_1", first.Choices[1].Delta.ToolCalls[0].ID)

	second, err := stream.Recv(context.Background())
	require.NoError(t, err)
	require.Equal(t, "call_0", second.Choices[0].Delta.ToolCalls[0].ID)
	require.Equal(t, `{"foo":1}`, second.Choices[0].Delta.ToolCalls[0].Function.Arguments)
	require.Equal(t, "call_1", second.Choices[1].Delta.ToolCalls[0].ID)
	require.Equal(t, `{"bar":2}`, second.Choices[1].Delta.ToolCalls[0].Function.Arguments)
}

func TestChatStream_Recv_NilStream(t *testing.T) {
	t.Parallel()

	stream := NewChatStream(nil)
	_, err := stream.Recv(context.Background())
	require.Error(t, err)

	var sdkErr *hferrors.SDKError
	require.ErrorAs(t, err, &sdkErr)
	require.Equal(t, hferrors.SDKErrorKindInternal, sdkErr.Kind)
}

func TestChatStream_Recv_EOF(t *testing.T) {
	t.Parallel()

	stream := NewChatStream(&mockChatStreamer{
		responses: []ChatStreamResponse{},
	})

	_, err := stream.Recv(context.Background())
	require.ErrorIs(t, err, io.EOF)
}

func TestChatStream_Close_NilStream(t *testing.T) {
	t.Parallel()

	stream := NewChatStream(nil)
	require.NoError(t, stream.Close())
}

func TestChatStream_Close_CallsUnderlying(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("close failed")
	mock := &mockChatStreamer{closeErr: wantErr}
	stream := NewChatStream(mock)

	err := stream.Close()
	require.ErrorIs(t, err, wantErr)
	require.True(t, mock.closed)
}
