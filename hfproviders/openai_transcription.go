package hfproviders

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"slices"
	"strconv"

	"github.com/Kardbord/hfgo/v4/hferrors"
	"github.com/Kardbord/hfgo/v4/hftypes"
)

// This file holds the wire-format helpers shared by every partner provider
// whose speech-recognition task is served through an OpenAI-compatible
// transcriptions endpoint proxied by the Hugging Face router (e.g. DeepInfra,
// Together). The codecs encode the canonical request as a multipart/form-data
// body and decode the endpoint's verbose_json response; the canonical model
// translation is provider-owned and stays in each provider's file.

// openAIFLACSignature is the magic prefix of a FLAC stream. net/http's byte
// sniffer has no FLAC signature, so the speech-recognition codecs check it
// explicitly to label FLAC input correctly.
const openAIFLACSignature = "fLaC"

// openAIDefaultAudioExtension is the filename extension used for audio whose
// media type is unknown, matching the official clients' fallback.
const openAIDefaultAudioExtension = "wav"

// openAISniffAudioType returns the media type of the given audio bytes,
// extending net/http's byte sniffer with an explicit FLAC check. Everything
// the sniffer cannot identify and that is not FLAC comes back as
// "application/octet-stream"; callers pass the result straight to
// [openAIMultipartBodyTo], whose extension table falls back to ".wav".
func openAISniffAudioType(audio []byte) string {
	sniffed := http.DetectContentType(audio)
	if sniffed == "application/octet-stream" &&
		bytes.HasPrefix(audio, []byte(openAIFLACSignature)) {
		return "audio/flac"
	}

	return sniffed
}

// openAIEncodeTranscription builds an OpenAI-compatible transcription
// endpoint's multipart/form-data body from the canonical request: the decoded
// audio rides the "file" part with its media type sniffed from the bytes'
// magic signature, the recognition and provider parameters ride the remaining
// form fields, and the mapped provider model identifier rides the "model"
// field — applied last, so provider parameters cannot override it. It returns
// the body and the Content-Type/Accept headers describing the wire format.
func openAIEncodeTranscription(
	req hftypes.SpeechRecognitionRequest,
	mapping *ModelMapping,
) (body []byte, headers http.Header, err error) {
	audio, err := base64.StdEncoding.DecodeString(req.Input)
	if err != nil {
		return nil, nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeBase64Input,
			Err:     err,
		}
	}

	// Sniff the audio's media type from its magic bytes: it drives both the
	// file part's Content-Type and the part's filename extension.
	sniffed := openAISniffAudioType(audio)

	fields, err := openAITranscriptionFields(&req)
	if err != nil {
		return nil, nil, err
	}
	// Applied last so a caller-supplied entry cannot override the mapped
	// provider model, matching the official clients.
	fields["model"] = mapping.ProviderModelID

	var bodyBuf bytes.Buffer
	contentType, err := openAIMultipartBodyTo(&bodyBuf, audio, sniffed, fields)
	if err != nil {
		return nil, nil, err
	}

	headers = http.Header{}
	headers.Set("Content-Type", contentType)
	headers.Set("Accept", mimeTypeJSON)

	return bodyBuf.Bytes(), headers, nil
}

// openAITranscriptionFields collects the canonical recognition parameters and
// provider parameters as form-field values, skipping nil provider entries.
// The mapped model is not part of the result: the caller applies it last so
// provider parameters cannot override it.
func openAITranscriptionFields(
	req *hftypes.SpeechRecognitionRequest,
) (map[string]string, error) {
	fields := map[string]string{}

	if req.Parameters != nil && req.Parameters.ReturnTimestamps != nil {
		fields["return_timestamps"] = strconv.FormatBool(*req.Parameters.ReturnTimestamps)
	}

	for key, value := range req.ProviderParams {
		if value == nil {
			continue
		}

		formatted, err := openAIFormFieldValue(value)
		if err != nil {
			return nil, err
		}

		fields[key] = formatted
	}

	return fields, nil
}

// openAITranscriptionSegment is one segment of an OpenAI-compatible
// transcription response.
type openAITranscriptionSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// openAIDecodeTranscription decodes an OpenAI-compatible transcription
// endpoint's JSON response into the canonical response: the transcript rides
// on "text" and segments map onto canonical chunks with [start, end] second
// timestamps. Every other top-level field (e.g. usage, duration, language)
// rides ProviderMetadata. A body without a string transcript is a
// serialization error.
func openAIDecodeTranscription(params DecodeParams) (hftypes.SpeechRecognitionResponse, error) {
	contentType := params.Headers.Get("Content-Type")
	if !isJSONContentType(contentType) {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "expected Content-Type application/json, got " + contentType,
			Err:     nil,
		}
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(params.Body, &fields); err != nil {
		return hftypes.SpeechRecognitionResponse{}, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	text, err := openAITranscriptionText(fields)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	var chunks []hftypes.SpeechRecognitionChunk
	if segmentsRaw, present := fields["segments"]; present && !isJSONNull(segmentsRaw) {
		chunks, err = openAITranscriptionChunks(segmentsRaw)
		if err != nil {
			return hftypes.SpeechRecognitionResponse{}, err
		}
	}

	metadata, err := openAITranscriptionMetadata(fields)
	if err != nil {
		return hftypes.SpeechRecognitionResponse{}, err
	}

	resp := hftypes.SpeechRecognitionResponse{
		Text:             text,
		Chunks:           chunks,
		ProviderMetadata: nil,
	}
	if len(metadata) > 0 {
		resp.ProviderMetadata = metadata
	}

	return resp, nil
}

// openAITranscriptionText extracts the response's transcript. A missing or
// non-string "text" field is a serialization error.
func openAITranscriptionText(fields map[string]json.RawMessage) (string, error) {
	textRaw, present := fields["text"]
	if !present || isJSONNull(textRaw) {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "transcription response missing text",
			Err:     nil,
		}
	}

	var text string
	if err := json.Unmarshal(textRaw, &text); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "transcription response text must be a string",
			Err:     err,
		}
	}

	return text, nil
}

// openAITranscriptionChunks maps the response's segments onto canonical chunks
// with [start, end] second timestamps. A non-array segments field is a
// serialization error; the caller leaves chunks unset when the field is
// absent or null.
func openAITranscriptionChunks(
	segmentsRaw json.RawMessage,
) ([]hftypes.SpeechRecognitionChunk, error) {
	var segments []openAITranscriptionSegment
	if err := json.Unmarshal(segmentsRaw, &segments); err != nil {
		return nil, &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToDecodeResponseBody,
			Err:     err,
		}
	}

	chunks := make([]hftypes.SpeechRecognitionChunk, 0, len(segments))
	for _, segment := range segments {
		chunks = append(chunks, hftypes.SpeechRecognitionChunk{
			Text:      segment.Text,
			Timestamp: []float64{segment.Start, segment.End},
		})
	}

	return chunks, nil
}

// openAITranscriptionMetadata collects every top-level response field other
// than the canonical ones (text, segments). The returned map may be empty
// when the response carries nothing beyond the canonical fields; the caller
// only attaches it when there is something in it.
func openAITranscriptionMetadata(
	fields map[string]json.RawMessage,
) (map[string]any, error) {
	metadata := make(map[string]any)
	for key, raw := range fields {
		if key == "text" || key == "segments" {
			continue
		}

		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: errFailedToDecodeResponseBody,
				Err:     err,
			}
		}

		metadata[key] = value
	}

	return metadata, nil
}

// openAIAudioExtension returns the filename extension for the given audio
// media type, defaulting to ".wav" for unknown or empty types. The mapping
// mirrors huggingface.js's media-type table, extended with the spellings
// net/http's byte sniffer actually emits: mp4 and webm containers are labeled
// video/*.
func openAIAudioExtension(mimeType string) string {
	mediaType, ok := parseMediaType(mimeType)
	if ok {
		switch mediaType {
		case "audio/mpeg", "audio/mp3":
			return "mp3"
		case "audio/mp4", "video/mp4":
			return "mp4"
		case "audio/m4a", "audio/x-m4a":
			return "m4a"
		case "audio/flac", "audio/x-flac":
			return "flac"
		case "audio/ogg", "application/ogg":
			return "ogg"
		case "audio/wav", "audio/wave", "audio/x-wav":
			return "wav"
		case "audio/webm", "video/webm":
			return "webm"
		}
	}

	return openAIDefaultAudioExtension
}

// openAIFormFieldValue formats one form-field value the way the official
// clients do: strings pass through unchanged, booleans and numbers stringify,
// and any other value JSON-encodes.
func openAIFormFieldValue(value any) (string, error) {
	switch typed := value.(type) {
	case string:
		return typed, nil
	case bool:
		return strconv.FormatBool(typed), nil
	case json.Number:
		return typed.String(), nil
	}

	if formatted, ok := openAIIntFormFieldValue(value); ok {
		return formatted, nil
	}
	if formatted, ok := openAIUintFormFieldValue(value); ok {
		return formatted, nil
	}
	if formatted, ok := openAIFloatFormFieldValue(value); ok {
		return formatted, nil
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: "failed to encode provider parameter as a form value",
			Err:     err,
		}
	}

	return string(encoded), nil
}

// openAIIntFormFieldValue stringifies signed integers of every width. ok is
// false for values that are not signed integers.
func openAIIntFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case int:
		return strconv.FormatInt(int64(typed), 10), true
	case int8:
		return strconv.FormatInt(int64(typed), 10), true
	case int16:
		return strconv.FormatInt(int64(typed), 10), true
	case int32:
		return strconv.FormatInt(int64(typed), 10), true
	case int64:
		return strconv.FormatInt(typed, 10), true
	}

	return "", false
}

// openAIUintFormFieldValue stringifies unsigned integers of every width. ok
// is false for values that are not unsigned integers.
func openAIUintFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case uint:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint8:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint16:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint32:
		return strconv.FormatUint(uint64(typed), 10), true
	case uint64:
		return strconv.FormatUint(typed, 10), true
	}

	return "", false
}

// openAIFloatFormFieldValue stringifies floats of both widths. ok is false
// for values that are not floats.
func openAIFloatFormFieldValue(value any) (string, bool) {
	switch typed := value.(type) {
	case float32:
		return strconv.FormatFloat(float64(typed), 'g', -1, 32), true
	case float64:
		return strconv.FormatFloat(typed, 'g', -1, 64), true
	}

	return "", false
}

// openAIMultipartBodyTo writes an OpenAI-compatible transcription endpoint's
// multipart body to dst: the audio rides on the "file" part — its
// Content-Type set to the sniffed media type and its filename derived from
// it — followed by the remaining form fields in sorted key order, matching
// the official clients' layout. It returns the body's Content-Type header
// value.
//
// dst exists purely as a test seam: production callers pass an in-memory
// buffer whose writes cannot fail, while tests pass a failing writer to
// exercise the defensive write-failure branches below.
func openAIMultipartBodyTo(
	dst io.Writer,
	audio []byte,
	mimeType string,
	fields map[string]string,
) (contentType string, err error) {
	writer := multipart.NewWriter(dst)

	partHeader := textproto.MIMEHeader{}
	partHeader.Set(
		"Content-Disposition",
		fmt.Sprintf(
			`form-data; name="file"; filename="audio.%s"`,
			openAIAudioExtension(mimeType),
		),
	)
	partHeader.Set("Content-Type", mimeType)
	part, err := writer.CreatePart(partHeader)
	if err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}
	if _, err := part.Write(audio); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	for _, key := range slices.Sorted(maps.Keys(fields)) {
		if err := writer.WriteField(key, fields[key]); err != nil {
			return "", &hferrors.SDKError{
				Kind:    hferrors.SDKErrorKindSerialization,
				Message: errFailedToMarshalRequestBody,
				Err:     err,
			}
		}
	}

	if err := writer.Close(); err != nil {
		return "", &hferrors.SDKError{
			Kind:    hferrors.SDKErrorKindSerialization,
			Message: errFailedToMarshalRequestBody,
			Err:     err,
		}
	}

	return writer.FormDataContentType(), nil
}
