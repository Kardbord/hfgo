# hfgo - HuggingFace Inference API Bruno Collection

A collection of [Bruno](https://www.usebruno.com/) requests for exercising the
HuggingFace Inference API (the `router.huggingface.co` endpoint used by the
hfgo Go SDK).

## Getting started

1. Open the Bruno **desktop app** and open this `hfgo` folder as a collection.
2. Select the `default` environment.
3. Switch the collection to **Developer Mode** (Shield icon, top right) - the
   pre-request scripts need `fetch`, which Safe Mode does not provide (see
   [Developer Mode](#developer-mode)).
4. Set the environment variables you need (see below) and run any request.

## Environment variables

Defined in `environments/`:

- `HF_TOKEN` - your HuggingFace access token, sent as `Authorization: Bearer`.
- `HF_BASE_URL` - the inference host (`https://router.huggingface.co`).
- `model_*` - the model used per task (one per task folder).

Requests use single, batch, and parametrized variants where the API supports
them.

## Binary sample data

No sample media is committed to this repository. The image/audio tasks fetch
their input at request time via a pre-request script and keep it purely in
memory (see [`scripts/samples.js`](./scripts/samples.js)):

- image tasks download a random photo from Lorem Picsum (`IMAGE_SAMPLE_URL`).
  Bruno's dynamic-variable image URLs are not used: the loremflickr-based ones
  return 401 and `$randomImageUrl`'s host varies by faker version.
- audio tasks download a public-domain speech sample (`jfk.flac` from OpenAI
  Whisper's test assets).

These downloads mean the image/audio requests need network access to
`picsum.photos` and `raw.githubusercontent.com` in addition to the HuggingFace
API.

## Developer Mode

The sample-download pre-request scripts use `fetch` and `Buffer`, which are
only available in Bruno's **developer** sandbox. Safe Mode is the default and
fails with `fetch is not defined`.

- **desktop app** - click the Shield icon in the top right corner and select
  **Developer Mode**.
- **CLI** - Safe Mode is the default since CLI v3, so pass
  `--sandbox=developer`.

## Binary request shapes

The scripts set bodies matching the HuggingFace Inference API contract:

- `*-single` (no `parameters`) - the sample bytes are sent as the raw request
  body (`req.setBody(bytes, { raw: true })`).
- `*-params` - JSON with the sample base64-encoded in `inputs` alongside a
  `parameters` object.
- `image-to-image` - JSON `inputs` holding a `data:` URI of the sample for both
  variants; the endpoint rejects raw-bytes bodies despite what its docs say.
- `image-text-to-text` - Chat Completion API (`/v1/chat/completions`) with the
  sample embedded as a `data:` URI in an `image_url` content part.

## Structure

- top-level folders - one per inference task
- `environments/` - environment configurations
- `scripts/` - shared pre-request helper (`samples.js`)
- `*.bru` - Bruno request files

## CLI (optional)

Requests can also be run with the Bruno CLI:

```
bruno-cli run --env default --sandbox=developer --output junit
```
