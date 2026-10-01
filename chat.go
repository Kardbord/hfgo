package hfgo

import (
	"github.com/Kardbord/hfgo/v4/internal/chatstream"
	"github.com/Kardbord/hfgo/v4/internal/dto"
)

//nolint:revive // type aliases forward docs from internal/dto
type (
	ChatRequest            = dto.ChatRequest
	ChatMessage            = dto.ChatMessage
	ChatMessageContent     = dto.ChatMessageContent
	ChatMessageChunk       = dto.ChatMessageChunk
	ChatImageURL           = dto.ChatImageURL
	ChatToolCall           = dto.ChatToolCall
	ChatFunctionDefinition = dto.ChatFunctionDefinition
	ChatResponseFormat     = dto.ChatResponseFormat
	ChatJSONSchemaConfig   = dto.ChatJSONSchemaConfig
	ChatStreamOptions      = dto.ChatStreamOptions
	ChatTool               = dto.ChatTool
	ChatToolChoice         = dto.ChatToolChoice
	ChatFunctionName       = dto.ChatFunctionName
	ToolChoiceMode         = dto.ToolChoiceMode
	ResponseFormatType     = dto.ResponseFormatType
	MessageChunkType       = dto.MessageChunkType
	ChatFunctionCall       = dto.ChatFunctionCall
)

//nolint:revive // type aliases forward docs from internal/dto
type (
	ChatResponse          = dto.ChatResponse
	ChatChoice            = dto.ChatChoice
	ChatLogProbs          = dto.ChatLogProbs
	ChatLogProb           = dto.ChatLogProb
	ChatTopLogProb        = dto.ChatTopLogProb
	ChatCompletionMessage = dto.ChatCompletionMessage
	ChatUsage             = dto.ChatUsage
	ChatToolCallOutput    = dto.ChatToolCallOutput
)

//nolint:revive // type aliases forward docs from internal/dto
type (
	ChatStreamResponse = dto.ChatStreamResponse
	ChatStreamChoice   = dto.ChatStreamChoice
	ChatStreamDelta    = dto.ChatStreamDelta
	ChatStreamToolCall = dto.ChatStreamToolCall
	ChatStreamFunction = dto.ChatStreamFunction
)

//nolint:revive // type aliases forward docs from internal/dto
const (
	MessageChunkTypeText         = dto.MessageChunkTypeText
	MessageChunkTypeImageURL     = dto.MessageChunkTypeImageURL
	ResponseFormatTypeText       = dto.ResponseFormatTypeText
	ResponseFormatTypeJSONSchema = dto.ResponseFormatTypeJSONSchema
	ResponseFormatTypeJSONObject = dto.ResponseFormatTypeJSONObject
	ToolChoiceModeAuto           = dto.ToolChoiceModeAuto
	ToolChoiceModeNone           = dto.ToolChoiceModeNone
	ToolChoiceModeRequired       = dto.ToolChoiceModeRequired
)

//nolint:revive // type aliases forward docs from internal/chatstream
type ChatStream = chatstream.ChatStream
