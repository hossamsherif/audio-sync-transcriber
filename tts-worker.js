/*
 * Local neural TTS Web Worker (Kokoro-82M via kokoro-js).
 *
 * Privacy-first text-to-speech that runs entirely in the browser — unlike the
 * Web Speech API, whose "natural/online" voices are synthesized server-side.
 * The main thread sends text + a voice id; we synthesize PCM here and transfer
 * it back for Web Audio playback.
 *
 * Model weights are fetched from the Hugging Face CDN and cached by
 * transformers.js (kokoro-js's dependency); the /+esm endpoint resolves
 * kokoro-js's bare imports. Spawned from the cross-origin isolated page
 * (sw.js), so multi-threaded WASM is available.
 */

import { KokoroTTS, env } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm';

// kokoro-js re-exports the same transformers.js `env` it uses internally, so we
// can tune its ONNX backend here. Mirror asr-worker.js: enable SIMD and, when
// cross-origin isolated (we are, via sw.js), give pure-WASM runs a real thread
// pool. Without this the WASM backend runs single-threaded — many times slower,
// which matters because WASM is the only correct path on GPUs whose WebGPU
// backend produces garbage audio (e.g. some brand-new AMD/RDNA parts).
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.simd = true;
}

// Must be set before the ONNX backend is first created (the pool size is read at
// init). WebGPU runs ops on the GPU, so a WASM thread pool would only add
// per-op barrier-sync overhead there — keep it single-threaded; for pure WASM
// use a capped pool without oversubscribing logical cores.
function configureOnnxThreads(device) {
  const wasm = env.backends?.onnx?.wasm;
  if (!wasm) return;
  if (!self.crossOriginIsolated) { wasm.numThreads = 1; return; }
  const cores = navigator.hardwareConcurrency || 4;
  wasm.numThreads = device === 'webgpu' ? 1 : Math.max(1, Math.min(cores - 1, 8));
}

let kokoro = null;
let currentKey = null;
// The generation request currently allowed to post results. A newer request or
// a 'cancel' invalidates the in-flight one between awaits.
let activeId = null;

function post(id, message, transfer) {
  self.postMessage({ ...message, id }, transfer || []);
}

async function getKokoro(id, model, device, dtype) {
  const key = `${model}__${device}__${dtype}`;
  if (kokoro && currentKey === key) return kokoro;
  configureOnnxThreads(device);
  kokoro = await KokoroTTS.from_pretrained(model, {
    dtype,
    device,
    progress_callback: (info) => post(id, { type: 'tts-progress', info }),
  });
  currentKey = key;
  return kokoro;
}

// Kokoro has a per-call token limit, so split long text on sentence boundaries
// into chunks under a character budget; audio is concatenated into one buffer.
function splitForTts(text, maxLen = 380) {
  const parts = String(text).match(/[^.!?…]+[.!?…]*\s*/g) || [String(text)];
  const chunks = [];
  let current = '';
  for (const part of parts) {
    if ((current + part).length > maxLen && current) {
      chunks.push(current.trim());
      current = part;
    } else {
      current += part;
    }
    while (current.length > maxLen) {
      chunks.push(current.slice(0, maxLen).trim());
      current = current.slice(maxLen);
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks.length ? chunks : [String(text).trim()];
}

async function generate(data) {
  const { id, text, voice, speed, model, device, dtype } = data;
  activeId = id;
  try {
    const tts = await getKokoro(id, model, device, dtype);
    if (activeId !== id) return;

    const chunks = splitForTts(text);
    const buffers = [];
    let sampleRate = 24000;
    let total = 0;
    for (const chunk of chunks) {
      if (activeId !== id) return;
      const raw = await tts.generate(chunk, { voice, speed });
      const arr = raw.audio instanceof Float32Array ? raw.audio : new Float32Array(raw.audio);
      sampleRate = raw.sampling_rate || sampleRate;
      buffers.push(arr);
      total += arr.length;
    }
    if (activeId !== id) return;

    const pcm = new Float32Array(total);
    let offset = 0;
    for (const buf of buffers) {
      pcm.set(buf, offset);
      offset += buf.length;
    }
    post(id, { type: 'tts-audio', pcm, sampleRate }, [pcm.buffer]);
  } catch (error) {
    if (activeId !== id) return;
    post(id, { type: 'tts-error', message: String((error && error.message) || error) });
  }
}

self.onmessage = (event) => {
  const data = event.data;
  if (!data) return;
  if (data.type === 'cancel') {
    if (activeId !== null) post(activeId, { type: 'tts-cancelled' });
    activeId = null;
    return;
  }
  if (data.type === 'generate') {
    if (activeId !== null && activeId !== data.id) post(activeId, { type: 'tts-cancelled' });
    generate(data);
  }
};
