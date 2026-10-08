// Shared sample-data helpers for the hfgo Bruno collection.
//
// Sample media is downloaded at request time and kept in memory so that no
// binary blobs are committed to the repository. Sample sources are pinned
// URLs (IMAGE_SAMPLE_URL / AUDIO_SAMPLE_URL) that callers pass to download.

// Public-domain speech sample (the classic JFK clip shipped with OpenAI
// Whisper's own test suite) used for the audio tasks.
const AUDIO_SAMPLE_URL = 'https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac';

// Random photo from Lorem Picsum for the image tasks. Bruno's dynamic-variable
// image URLs (loremflickr.com) return 401, so a pinned URL is used instead.
// Unseeded: a different photo is served on every request.
const IMAGE_SAMPLE_URL = 'https://picsum.photos/512/512';

// download fetches url and returns its body as a Buffer along with the
// response media type. Throws on non-2xx responses.
// Requires the developer sandbox: Safe Mode has no fetch (see README.md).
async function download(url) {
  if (bru.isSafeMode()) {
    throw new Error('sample downloads require Bruno Developer Mode (Safe Mode has no fetch); see README.md');
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`sample download failed: ${res.status} ${url}`);
  }
  const contentTypeHeader = typeof res.headers?.get === 'function' ? res.headers.get('content-type') : '';
  const contentType = (contentTypeHeader || 'application/octet-stream').split(';')[0].trim();
  const bytes = Buffer.from(await res.arrayBuffer());
  return { bytes, contentType };
}

// toBase64 returns the base64 form of a downloaded sample, as used by the
// "inputs" JSON payload.
function toBase64(sample) {
  return sample.bytes.toString('base64');
}

// toDataUri returns the data: URI form of a downloaded sample, for multimodal
// chat content.
function toDataUri(sample) {
  return `data:${sample.contentType};base64,${toBase64(sample)}`;
}

module.exports = {
  AUDIO_SAMPLE_URL,
  IMAGE_SAMPLE_URL,
  download,
  toBase64,
  toDataUri
};
