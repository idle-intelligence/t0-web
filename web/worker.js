/**
 * Web Worker: loads the t0-wasm module and model, runs every forecast.
 *
 * All inference runs here, never on the main thread (see this repo's
 * CLAUDE.md Demo section).
 *
 * Protocol:
 *   Main -> Worker:
 *     { type: 'load' }                       -- fetch WASM + model + series, init
 *     { type: 'forecast', origin: number }   -- forecast from this split point
 *
 *   Worker -> Main:
 *     { type: 'status', text: string }
 *     { type: 'ready', series, modelBytes, loadMs, nQuantiles, contextCap, horizon }
 *     { type: 'forecast', origin, quantiles, nQuantiles, ms }
 *     { type: 'error', message: string }
 */

const MODEL_URL = new URL('./models/t0-alpha-q8_0.gguf', import.meta.url).href;
const SERIES_URL = new URL('./data/series.f32', import.meta.url).href;
const CACHE_NAME = 't0-model-v1';

// Context is capped at the same 512 used by docs/BENCHMARKS.md's latency
// table; horizon 32 matches it too, so this demo's per-forecast ms is
// directly comparable to that native CPU number.
const CONTEXT_CAP = 512;
const HORIZON = 32;

let t0wasm = null;
let model = null;
let series = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;
    try {
        if (type === 'load') {
            await handleLoad();
        } else if (type === 'forecast') {
            handleForecast(data.origin);
        } else {
            console.warn('[worker] unknown message type:', type);
        }
    } catch (err) {
        self.postMessage({ type: 'error', message: err.message || String(err) });
    }
};

async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
        self.postMessage({ type: 'status', text: `${label} (cached)` });
        return await cached.arrayBuffer();
    }

    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`fetch ${url}: ${resp.status} ${resp.statusText}`);
    }
    const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        if (contentLength > 0) {
            const pct = ((loaded / contentLength) * 100).toFixed(0);
            self.postMessage({ type: 'status', text: `${label}: ${pct}%` });
        }
    }
    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buf.set(chunk, offset);
        offset += chunk.byteLength;
    }
    try {
        await cache.put(url, new Response(buf.buffer, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } catch (cacheErr) {
        console.warn('[worker] could not cache:', cacheErr);
    }
    return buf.buffer;
}

async function handleLoad() {
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    const wasmJsUrl = new URL('./pkg/t0_wasm.js', import.meta.url).href;
    t0wasm = await import(wasmJsUrl);
    await t0wasm.default();

    self.postMessage({ type: 'status', text: 'Downloading model (Q8_0, ~109 MB)...' });
    const modelBuf = await cachedFetch(MODEL_URL, 'Downloading model');

    self.postMessage({ type: 'status', text: 'Loading model...' });
    const t0 = performance.now();
    model = t0wasm.T0Wasm.load(new Uint8Array(modelBuf));
    const loadMs = performance.now() - t0;

    self.postMessage({ type: 'status', text: 'Loading series...' });
    const seriesBuf = await fetch(SERIES_URL).then((r) => r.arrayBuffer());
    series = new Float32Array(seriesBuf);

    self.postMessage({
        type: 'ready',
        series: Array.from(series),
        modelBytes: modelBuf.byteLength,
        loadMs,
        nQuantiles: model.nQuantiles(),
        contextCap: CONTEXT_CAP,
        horizon: HORIZON,
    });
    self.postMessage({ type: 'status', text: 'Ready' });
}

function handleForecast(origin) {
    if (!model || !series) {
        self.postMessage({ type: 'error', message: 'forecast requested before model/series ready' });
        return;
    }
    const ctxStart = Math.max(0, origin - CONTEXT_CAP);
    const context = series.slice(ctxStart, origin);
    const t0 = performance.now();
    const quantiles = model.forecast(context, HORIZON);
    const ms = performance.now() - t0;
    self.postMessage({
        type: 'forecast',
        origin,
        quantiles: Array.from(quantiles),
        nQuantiles: model.nQuantiles(),
        horizon: HORIZON,
        ms,
    });
}
