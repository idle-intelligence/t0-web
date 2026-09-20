/**
 * Web Worker: loads the t0-wasm module and model, runs every forecast.
 *
 * All inference runs here, never on the main thread.
 *
 * Protocol:
 *   Main -> Worker:
 *     { type: 'load', modelKey } -- unload any previous model, fetch WASM
 *                                    (once) + the chosen model + config +
 *                                    series (once), init
 *     { type: 'forecast', origin: number, requestId: number } -- forecast from this split point
 *
 *   Worker -> Main:
 *     { type: 'models', list: {key, label}[], defaultKey } -- sent once at startup
 *     { type: 'status', text: string, key?: string }  -- key present => update that one log line in place
 *     { type: 'ready', modelKey, series, seriesName, startDate, freq, modelBytes, loadMs, nQuantiles, quantileLevels, contextCap, horizon, backend }
 *     { type: 'forecast', origin, originDate, requestId, quantiles, nQuantiles, horizon, ms }
 *     { type: 'error', message: string }
 */

// Set to e.g. './models' to load `<LOCAL_MODELS_DIR>/<key>/<file>` and
// `<LOCAL_MODELS_DIR>/<key>/config.json` instead of Hugging Face -- lets the
// four weights be tested before their HF repos exist. Empty by default.
const LOCAL_MODELS_DIR = '';

// Version tag on the engine URLs: browsers cache the wasm at a fixed path
// across rebuilds, even through a hard reload. Bump when the engine changes.
const ENGINE_BUILD = '2026-09-21';

// The four released weights. Each Hugging Face repo holds one GGUF plus its
// config.json (the source of that model's quantile levels).
const MODELS = {
    'alpha-q8': { label: 'alpha Q8_0 · 109 MB', hfRepo: 'idle-intelligence/t0-alpha-q8_0-webgpu', file: 't0-alpha-q8_0.gguf' },
    'alpha-q4': { label: 'alpha Q4_0 · 59 MB', hfRepo: 'idle-intelligence/t0-alpha-q4_0-webgpu', file: 't0-alpha-q4_0.gguf' },
    'beta-q8': { label: 'beta Q8_0 · 275 MB', hfRepo: 'idle-intelligence/t0-beta-q8_0-webgpu', file: 't0-beta-q8_0.gguf' },
    'beta-q4': { label: 'beta Q4_0 · 149 MB', hfRepo: 'idle-intelligence/t0-beta-q4_0-webgpu', file: 't0-beta-q4_0.gguf' },
};
const DEFAULT_MODEL_KEY = 'alpha-q4';

// Single source of truth for the model row: the main thread renders its
// buttons from this list rather than keeping its own copy.
self.postMessage({
    type: 'models',
    list: Object.entries(MODELS).map(([key, m]) => ({ key, label: m.label })),
    defaultKey: DEFAULT_MODEL_KEY,
});

function modelUrls(key) {
    const m = MODELS[key];
    if (LOCAL_MODELS_DIR) {
        return {
            gguf: new URL(`${LOCAL_MODELS_DIR}/${key}/${m.file}`, import.meta.url).href,
            config: new URL(`${LOCAL_MODELS_DIR}/${key}/config.json`, import.meta.url).href,
        };
    }
    return {
        gguf: `https://huggingface.co/${m.hfRepo}/resolve/main/${m.file}`,
        config: `https://huggingface.co/${m.hfRepo}/resolve/main/config.json`,
    };
}

const SERIES_URL = new URL('./data/series.f32', import.meta.url).href;
const SERIES_META_URL = new URL('./data/series_meta.json', import.meta.url).href;
const CACHE_NAME = 't0-model-v1';

// Context is capped at the same 512 used by docs/BENCHMARKS.md's latency
// table; horizon 32 matches it too, so this demo's per-forecast ms is
// directly comparable to that native CPU number. The chart only ever
// *displays* the trailing 160 context points (see index.html) -- the model
// still sees up to CONTEXT_CAP points.
const CONTEXT_CAP = 512;
const HORIZON = 32;

let t0wasm = null;
let model = null;
let series = null;
let seriesMeta = null;
let forecastInFlight = null; // Promise<Float32Array> of model.forecast() while it's pending

// freq is always 'D' for this bundled series (us_births) -- date-per-index
// is start_date + index days, no calendar-skip frequencies supported here.
function dateAtIndex(i) {
    const start = new Date(seriesMeta.start_date.replace(' ', 'T') + 'Z');
    const d = new Date(start.getTime() + i * 86400000);
    return d.toISOString().slice(0, 10);
}

self.onmessage = async (e) => {
    const { type, ...data } = e.data;
    try {
        if (type === 'load') {
            await handleLoad(data.modelKey);
        } else if (type === 'forecast') {
            await handleForecast(data.origin, data.requestId);
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
        self.postMessage({ type: 'status', key: 'download', text: `${label} (cached)` });
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
            self.postMessage({ type: 'status', key: 'download', text: `${label}: ${pct}%` });
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

// Backend selection: WebGPU if this worker's `navigator.gpu` exists AND an
// adapter can actually be obtained (dedicated workers get their own
// WorkerNavigator with the same `gpu` property as the window) -- some
// browsers/flags expose `navigator.gpu` but fail requestAdapter(), and
// t0-wasm's wgpu build has no graceful fallback for that (it traps), so
// the check has to be the real thing, not just feature detection. Two
// separate wasm-pack outputs -- `pkg-wgpu/` (built with
// `--no-default-features --features wgpu`) and `pkg/` (default,
// `ndarray`) -- since the backend is a compile-time Cargo feature, same
// convention as `crates/cli`.
let BACKEND = null;
let PKG_DIR = null;

async function detectBackend() {
    if (BACKEND) return;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                BACKEND = 'webgpu';
                PKG_DIR = './pkg-wgpu';
                return;
            }
        } catch (err) {
            // fall through to CPU
        }
    }
    BACKEND = 'wasm/ndarray';
    PKG_DIR = './pkg';
}

async function handleLoad(modelKey) {
    const key = MODELS[modelKey] ? modelKey : DEFAULT_MODEL_KEY;
    const m = MODELS[key];

    // A forecast may still be in flight (async GPU readback on wgpu) when a
    // model switch arrives -- wait for it before freeing the model out from
    // under it, or wasm-bindgen panics ("attempted to take ownership of
    // Rust value while it was borrowed").
    if (forecastInFlight) {
        await forecastInFlight.catch(() => {});
    }

    if (model) {
        model.free();
        model = null;
    }

    await detectBackend();

    if (!t0wasm) {
        self.postMessage({ type: 'status', text: `Loading WASM module (${BACKEND})...` });
        const wasmJsUrl = new URL(`${PKG_DIR}/t0_wasm.js?v=${ENGINE_BUILD}`, import.meta.url).href;
        t0wasm = await import(wasmJsUrl);
        await t0wasm.default({ module_or_path: new URL(`${PKG_DIR}/t0_wasm_bg.wasm?v=${ENGINE_BUILD}`, import.meta.url).href });
        // Must run before T0Wasm.load(): on wgpu this drives the async
        // requestAdapter()/requestDevice() setup that WASM has no blocking
        // executor for (see t0-wasm's initBackend doc comment); a no-op on
        // ndarray.
        await t0wasm.initBackend();
    }

    const urls = modelUrls(key);
    self.postMessage({ type: 'status', key: 'download', text: `Downloading ${m.label}...` });
    const modelBuf = await cachedFetch(urls.gguf, `Downloading ${m.label}`);

    self.postMessage({ type: 'status', key: 'load', text: 'Loading model...' });
    const t0 = performance.now();
    model = t0wasm.T0Wasm.load(new Uint8Array(modelBuf));
    const loadMs = performance.now() - t0;
    self.postMessage({ type: 'status', key: 'load', text: `Model loaded in ${loadMs.toFixed(0)} ms` });

    const config = await fetch(urls.config).then((r) => r.json());
    const quantileLevels = config.quantile_levels;

    if (!series) {
        self.postMessage({ type: 'status', key: 'series', text: 'Loading series...' });
        const seriesBuf = await fetch(SERIES_URL).then((r) => r.arrayBuffer());
        series = new Float32Array(seriesBuf);
        seriesMeta = await fetch(SERIES_META_URL).then((r) => r.json());
        self.postMessage({
            type: 'status',
            key: 'series',
            text: `Series: ${seriesMeta.name} (${seriesMeta.n_points} points, ${seriesMeta.start_date.slice(0, 10)} to ${dateAtIndex(seriesMeta.n_points - 1)})`,
        });
    }

    self.postMessage({ type: 'status', key: 'backend', text: `Backend: ${BACKEND}` });

    self.postMessage({
        type: 'ready',
        modelKey: key,
        series: Array.from(series),
        seriesName: seriesMeta.name,
        startDate: seriesMeta.start_date,
        freq: seriesMeta.freq,
        modelBytes: modelBuf.byteLength,
        loadMs,
        nQuantiles: model.nQuantiles(),
        quantileLevels,
        contextCap: CONTEXT_CAP,
        horizon: HORIZON,
        backend: BACKEND,
    });
}

async function handleForecast(origin, requestId) {
    if (!model || !series) {
        self.postMessage({ type: 'error', message: 'forecast requested before model/series ready' });
        return;
    }
    const ctxStart = Math.max(0, origin - CONTEXT_CAP);
    const context = series.slice(ctxStart, origin);
    const t0 = performance.now();
    // model.forecast is always async now (t0-wasm's wgpu build needs a real
    // async GPU readback; the ndarray build resolves the same Promise
    // immediately) -- always `await`, never assume a sync return value.
    // Tracked in forecastInFlight so a concurrent model switch waits for it
    // instead of freeing the model out from under this call.
    const p = model.forecast(context, HORIZON);
    forecastInFlight = p;
    let quantiles;
    try {
        quantiles = await p;
    } finally {
        if (forecastInFlight === p) forecastInFlight = null;
    }
    const ms = performance.now() - t0;
    self.postMessage({
        type: 'forecast',
        origin,
        originDate: dateAtIndex(origin),
        requestId,
        quantiles: Array.from(quantiles),
        nQuantiles: model.nQuantiles(),
        horizon: HORIZON,
        ms,
    });
}
