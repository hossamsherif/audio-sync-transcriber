/*
 * Service worker for Browser Audio Studio.
 *
 * Two responsibilities, both required because GitHub Pages cannot send custom
 * HTTP response headers:
 *
 *  1. Cross-origin isolation (COI shim). Injects COOP/COEP onto our own
 *     navigation + same-origin responses so the page becomes
 *     `crossOriginIsolated`, which unlocks SharedArrayBuffer and therefore
 *     multi-threaded WASM for onnx-runtime / transformers.js. We use COEP
 *     `credentialless` (not `require-corp`) so cross-origin model/library
 *     fetches from the Hugging Face and jsDelivr CDNs keep working without
 *     needing CORP headers they don't send. Modeled on the well-known
 *     `coi-serviceworker` project.
 *
 *  2. Offline app shell. Caches the page itself and the CDN ESM library so the
 *     app loads with no network. Model weights are intentionally NOT cached
 *     here: transformers.js already persists them via `env.useBrowserCache`.
 */

const CACHE_VERSION = 'bas-v2';
const APP_SHELL = ['./', './index.html'];

// Cross-origin hosts whose responses we cache at runtime (library + wasm).
// Kept as a prefix list so versioned CDN paths match.
const RUNTIME_CACHE_HOSTS = ['https://cdn.jsdelivr.net/', 'https://cdn.skypack.dev/', 'https://esm.sh/'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_VERSION).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

// Add the COOP/COEP headers that make the page cross-origin isolated.
function withCoiHeaders(response) {
  if (!response || response.status === 0) return response; // opaque, leave as-is
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function isRuntimeCacheable(url) {
  return RUNTIME_CACHE_HOSTS.some((host) => url.startsWith(host));
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = request.url;
  const isSameOrigin = url.startsWith(self.location.origin);

  // Same-origin (our app shell + sw-registered page): cache-first, then add COI
  // headers so navigations make the page cross-origin isolated.
  if (isSameOrigin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return withCoiHeaders(cached);
        return fetch(request)
          .then((response) => {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
            return withCoiHeaders(response);
          })
          .catch(() => cached);
      }),
    );
    return;
  }

  // Cross-origin CDN library + wasm: cache-first for offline. Do not rewrite
  // headers (credentialless lets these load without CORP).
  if (isRuntimeCacheable(url)) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response && (response.ok || response.type === 'opaque')) {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
          }
          return response;
        });
      }),
    );
  }
  // Everything else (e.g. HF weight CDN): pass through; transformers.js owns
  // its own caching.
});
