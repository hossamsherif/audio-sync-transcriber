# Browser Audio Studio

A static, browser-first audio utility for transcribing local audio files, syncing transcripts to playback, exporting captions, and reading text aloud with browser voices.

The app is designed to run from GitHub Pages or any normal HTTPS static host. Audio files stay local to the user's browser; they do not need to be uploaded into the repository.

## Current Features

- In-browser Whisper transcription powered by `@huggingface/transformers`
- Model selection for browser-friendly Whisper variants
- Device selection for `auto`, `webgpu`, or `wasm`
- Weight selection for `fp32` or `q8`
- Adjustable chunk length and stride for long audio
- Segment or word timestamp mode
- Audio player with synced, clickable transcript segments
- Active transcript highlighting during playback
- Transcript search and optional auto-scroll
- Download formats: JSON, SRT, VTT, and TXT
- Text-to-speech tab using the browser Web Speech API
- TTS word highlighting while browser speech is playing
- Searchable browser voice picker with voice reload
- Read all text, read selected text, or load the generated transcript into TTS
- TTS controls for speed, pitch, volume, pause, resume, and stop
- Light/dark theme toggle saved in browser storage
- Optional debug logging with `?debug=1`

## Use

1. Open the site.
2. Choose an audio file from your device.
3. Select the transcription model, device, weights, chunk settings, and timestamp mode if needed.
4. Click **Transcribe in browser**.
5. Play the audio and click transcript segments to jump through the file.
6. Search the transcript or download it as JSON, SRT, VTT, or TXT.
7. Open **Text to speech** to read pasted text, a selected passage, or the generated transcript aloud.

The first transcription run downloads the selected model into the browser cache. Later runs can reuse the cached model when the browser keeps that cache available.

## Browser Requirements

- Use a modern Chromium, Safari, or Firefox browser from an HTTPS page.
- WebGPU can improve performance when available. The app falls back to WASM when WebGPU is unavailable or when `wasm` is selected.
- Text-to-speech depends on the browser's built-in Web Speech API and installed/available voices, so voice availability varies by browser and operating system.

## Audio Limits

The app transcribes in the browser, so file size and duration limits are intentionally conservative:

- Desktop: up to 100 MB and 45 minutes
- Mobile: up to 40 MB and 15 minutes

Longer files should be split before transcription. Smaller Whisper models and `fp32` weights are generally the safest browser options.

## Publish on GitHub Pages

1. Create a GitHub repository.
2. Put `index.html`, `README.md`, `LICENSE`, and `.nojekyll` at the repository root.
3. In GitHub, open **Settings -> Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select `main` and `/(root)`.
6. Save, then open the generated Pages URL.

## Notes

- The app is a static site. There is no build step or backend service.
- Provider TTS engines and local Piper-style models are not wired into this GitHub Pages version because they require extra runtime or backend support.
- Quantized `q8` weights may fail in some browser/runtime combinations. Switch back to `fp32` if model session creation fails.
- Enable debug logging by opening the app with `?debug=1`.
