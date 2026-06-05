/*
 * ASR Web Worker.
 *
 * Runs Whisper transcription off the main thread so the UI never blocks. The
 * main thread decodes audio (Web Audio's AudioContext is main-thread only),
 * transfers the 16 kHz mono Float32Array here, and we stream timestamped
 * segments back chunk-by-chunk.
 *
 * This worker owns everything compute-related: the transformers.js pipeline,
 * its env config, the memoized transcriber, and the pure chunking/alignment
 * helpers. The main thread keeps DOM, state, and the readable-segment merge.
 *
 * Because this worker is spawned from a cross-origin-isolated page (see sw.js),
 * it is itself cross-origin isolated, so multi-threaded WASM is available.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowRemoteModels = true;
env.useBrowserCache = true;
env.allowLocalModels = false;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.simd = true;
}

// Pick the WASM thread count for the device this run will use. Must be set
// before the ONNX backend is first created (the pool size is read at init).
// WebGPU runs most ops on the GPU; only a few shape/control ops fall back to
// WASM, and Whisper decodes autoregressively (many tiny CPU ops per token), so
// any thread pool just adds per-op SharedArrayBuffer barrier-sync overhead —
// worse on Apple Silicon, where barriers stall fast P-cores on E-cores. So keep
// WebGPU single-threaded; for pure WASM use a capped pool (a real speedup
// without oversubscribing logical cores).
function configureOnnxThreads(device) {
  const wasm = env.backends?.onnx?.wasm;
  if (!wasm) return;
  if (!self.crossOriginIsolated) { wasm.numThreads = 1; return; }
  const cores = navigator.hardwareConcurrency || 4;
  wasm.numThreads = device === 'webgpu' ? 1 : Math.max(1, Math.min(cores - 1, 8));
}

const TARGET_SAMPLE_RATE = 16000;

// Set per job from the main thread (used to fill missing segment end times,
// replacing the main thread's audioEl.duration reference).
let audioDuration = 0;
let transcriber = null;
let currentKey = null;
// The job currently allowed to post results. A newer job or a 'cancel' message
// invalidates the in-flight loop between awaits.
let activeJob = null;

function post(job, message, transfer) {
  self.postMessage({ ...message, job }, transfer || []);
}

// --- Pure helpers (moved verbatim from index.html, audioEl.duration -> audioDuration) ---

function normalizeSegments(rawSegments) {
  const normalized = rawSegments
    .map(seg => ({
      start: Number(seg.start),
      end: seg.end == null ? null : Number(seg.end),
      text: String(seg.text || '').trim(),
    }))
    .filter(seg => Number.isFinite(seg.start) && seg.text)
    .sort((a, b) => a.start - b.start);

  return normalized.map((seg, idx, arr) => {
    let end = seg.end;
    const nextStart = arr[idx + 1]?.start;
    if (!(Number.isFinite(end) && end > seg.start)) {
      if (Number.isFinite(nextStart) && nextStart > seg.start) end = nextStart;
      else if (Number.isFinite(audioDuration) && audioDuration > seg.start) end = audioDuration;
      else end = seg.start + 2;
    }
    return { ...seg, end };
  });
}

function groupWordChunks(chunks) {
  const rows = [];
  let current = null;
  let count = 0;
  for (const chunk of chunks) {
    const [start, end] = Array.isArray(chunk.timestamp) ? chunk.timestamp : [null, null];
    const text = String(chunk.text || '').trim();
    if (!Number.isFinite(start) || !text) continue;
    if (!current) {
      current = { start, end: Number.isFinite(end) ? end : start + 0.8, text };
      count = 1;
      continue;
    }
    current.end = Number.isFinite(end) ? end : current.end;
    current.text += (text.startsWith("'") || text.startsWith(',') || text.startsWith('.') || text.startsWith('!') || text.startsWith('?') ? '' : ' ') + text;
    count += 1;
    const duration = current.end - current.start;
    const shouldBreak = /[.!?…:]$/.test(text) || duration >= 6 || count >= 14;
    if (shouldBreak) {
      rows.push(current);
      current = null;
      count = 0;
    }
  }
  if (current) rows.push(current);
  return rows;
}

function outputToSegments(output, timestampMode) {
  if (Array.isArray(output?.chunks) && output.chunks.length) {
    if (timestampMode === 'word') {
      return normalizeSegments(groupWordChunks(output.chunks));
    }
    return normalizeSegments(output.chunks.map(chunk => ({
      start: Array.isArray(chunk.timestamp) ? chunk.timestamp[0] : 0,
      end: Array.isArray(chunk.timestamp) ? chunk.timestamp[1] : null,
      text: chunk.text,
    })));
  }
  if (typeof output?.text === 'string' && output.text.trim()) {
    return normalizeSegments([{ start: 0, end: audioDuration || 0, text: output.text.trim() }]);
  }
  return [];
}

function getChunkWindows(totalSamples, chunkLengthSec, strideSec) {
  const chunkSize = Math.max(1, Math.round(chunkLengthSec * TARGET_SAMPLE_RATE));
  const overlap = Math.max(0, Math.round(strideSec * TARGET_SAMPLE_RATE));
  const totalDuration = totalSamples / TARGET_SAMPLE_RATE;
  const windows = [];
  for (let start = 0; start < totalSamples; start += chunkSize) {
    const centralStart = start / TARGET_SAMPLE_RATE;
    const centralEnd = Math.min(totalDuration, centralStart + chunkLengthSec);
    const windowStart = Math.max(0, Math.round(centralStart * TARGET_SAMPLE_RATE) - overlap);
    const windowEnd = Math.min(
      totalSamples,
      Math.round(centralEnd * TARGET_SAMPLE_RATE) + overlap,
    );
    windows.push({
      centralStart,
      centralEnd,
      windowStartSec: windowStart / TARGET_SAMPLE_RATE,
      chunkAudio: audioData => audioData.subarray(windowStart, windowEnd),
    });
  }
  return windows;
}

function alignChunkSegments(rawSegments, windowStartSec, centralStart, centralEnd) {
  return rawSegments
    .map(seg => ({
      start: seg.start + windowStartSec,
      end: seg.end + windowStartSec,
      text: seg.text,
    }))
    .filter(seg => Number.isFinite(seg.start) && Number.isFinite(seg.end))
    .filter(seg => seg.end > centralStart && seg.start < centralEnd)
    .map(seg => ({
      ...seg,
      start: Math.max(seg.start, centralStart),
      end: Math.min(seg.end, centralEnd),
    }))
    .filter(seg => seg.end > seg.start && seg.text);
}

async function getTranscriber(job, model, device, dtype) {
  const key = `${model}__${device}__${dtype}`;
  if (transcriber && currentKey === key) return transcriber;
  transcriber = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: (info) => post(job, { type: 'model-progress', info }),
  });
  currentKey = key;
  return transcriber;
}

// --- Job loop ---

async function runTranscription(data) {
  const { job, audio, model, device, dtype, chunkLength, strideLength, timestampMode, debug } = data;
  activeJob = job;
  audioDuration = data.audioDuration || 0;
  const audioData = audio; // Float32Array over the transferred buffer

  try {
    configureOnnxThreads(device); // before the backend is first created
    const asr = await getTranscriber(job, model, device, dtype);
    if (activeJob !== job) return;
    post(job, { type: 'model-ready' });

    const chunkWindows = getChunkWindows(audioData.length, chunkLength, strideLength);
    const totalChunks = Math.max(1, chunkWindows.length);
    for (let i = 0; i < totalChunks; i++) {
      if (activeJob !== job) return;
      const chunkStart = performance.now();
      const { centralStart, centralEnd, windowStartSec, chunkAudio } = chunkWindows[i];
      if (debug) post(job, { type: 'debug', message: `chunk ${i + 1}/${totalChunks}: window [${centralStart.toFixed(2)}, ${centralEnd.toFixed(2)}], strideWindowStart=${windowStartSec.toFixed(2)}` });
      try {
        const windowData = chunkAudio(audioData);
        const asrStart = performance.now();
        const output = await asr(windowData, {
          return_timestamps: timestampMode === 'word' ? 'word' : true,
        });
        if (debug) post(job, { type: 'debug', message: `chunk ${i + 1}/${totalChunks} ASR time ${(performance.now() - asrStart).toFixed(1)} ms` });
        if (activeJob !== job) return;
        const chunkSegments = alignChunkSegments(
          outputToSegments(output, timestampMode),
          windowStartSec,
          centralStart,
          centralEnd,
        );
        if (debug) post(job, { type: 'debug', message: `chunk ${i + 1}/${totalChunks} aligned ${chunkSegments.length} segments` });
        post(job, { type: 'chunk', segments: chunkSegments, chunkIndex: i + 1, totalChunks, centralEnd });
        if (debug) post(job, { type: 'debug', message: `chunk ${i + 1}/${totalChunks} loop wall time ${(performance.now() - chunkStart).toFixed(1)} ms` });
      } catch (err) {
        post(job, { type: 'chunk-error', chunkIndex: i + 1, totalChunks, message: String((err && err.message) || err) });
      }
    }
    if (activeJob !== job) return;
    post(job, { type: 'done', totalChunks });
  } catch (error) {
    if (activeJob !== job) return;
    post(job, { type: 'error', message: String((error && error.message) || error) });
  }
}

self.onmessage = (event) => {
  const data = event.data;
  if (!data) return;
  if (data.type === 'cancel') {
    // Settle any in-flight job's promise on the main thread.
    if (activeJob !== null) post(activeJob, { type: 'cancelled' });
    activeJob = null;
    return;
  }
  if (data.type === 'transcribe') {
    // A new job preempts the previous one; settle its promise too.
    if (activeJob !== null && activeJob !== data.job) post(activeJob, { type: 'cancelled' });
    runTranscription(data);
  }
};
