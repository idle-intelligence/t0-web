// bench/compare/compare.js
//
// Single-page head-to-head: theforecastingcompany's t0-alpha-onnx-int8
// export (onnxruntime-web, WebGPU EP with wasm fallback) vs our GGUF-
// resident t0-fast (Q8_0/Q4_0, crates/t0-wasm/pkg-fast). Both engines load
// in this same tab so the numbers are for whatever machine opens this page.
//
// Driven either by the UI (run button) or headlessly via window.__app.run(),
// per this repo's rule of driving the demo through a small API rather than
// DOM clicks in Playwright.

// Set to true to use the gitignored bench/onnx and bench/ours folders on
// disk instead of Hugging Face and the CDN. Off by default: this is the
// mode that gets published to GitHub Pages.
const LOCAL_ASSETS = false;

// Matches the pinned onnxruntime-web build committed at bench/onnx/ (see
// its "ONNX Runtime Web v1.29.0" banner) and the manifest.json served
// alongside the official INT8 export.
const ORT_VERSION = '1.29.0';
const ORT_CDN_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_SCRIPT_URL = LOCAL_ASSETS ? '../onnx/ort.webgpu.min.js' : `${ORT_CDN_BASE}ort.webgpu.min.js`;
const ORT_WASM_PATHS = LOCAL_ASSETS ? '../onnx/' : ORT_CDN_BASE;

// Same engine-build version tag web/worker.js uses on its wasm URLs, so a
// hard reload here picks up a rebuilt pkg-wgpu the same way the demo does.
const ENGINE_BUILD = '2026-09-21';

const ONNX_MODEL_URL = LOCAL_ASSETS
    ? '../onnx/t0-alpha-grouped-int8.onnx'
    : 'https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8/resolve/main/t0-alpha-grouped-int8.onnx';
const ONNX_SERIES_URL = LOCAL_ASSETS ? '../onnx/series.f32' : '../../web/data/series.f32';
const OURS_PKG = LOCAL_ASSETS ? '../ours/pkg-fast/t0_wasm.js' : `../../web/pkg-wgpu/t0_wasm.js?v=${ENGINE_BUILD}`;
const OURS_GGUF = LOCAL_ASSETS
    ? { q8_0: '../ours/t0-alpha-q8_0.gguf', q4_0: '../ours/t0-alpha-q4_0.gguf' }
    : {
        q8_0: 'https://huggingface.co/idle-intelligence/t0-alpha-q8_0-webgpu/resolve/main/t0-alpha-q8_0.gguf',
        q4_0: 'https://huggingface.co/idle-intelligence/t0-alpha-q4_0-webgpu/resolve/main/t0-alpha-q4_0.gguf',
    };
// F32 reference is opt-in (407 MB download) -- see the checkbox wiring below.
const OURS_F32_SAFETENSORS = LOCAL_ASSETS
    ? '../ours/t0-alpha-f32.safetensors'
    : 'https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/model.safetensors';
const OURS_F32_CONFIG = LOCAL_ASSETS
    ? '../ours/t0-alpha-f32-config.json'
    : 'https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/config.json';

const CH_OPTIONS = [
    { context: 512, horizon: 32 },
    { context: 512, horizon: 128 },
    { context: 256, horizon: 64 },
    { context: 2048, horizon: 128 },
];
const ORIGIN = 3600; // same fixture window as bench/ours, bench/onnx, web/worker.js
const N_BATCH = 24;
const N_WARM = 10;
const QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90];
const NQ = QUANTILE_LEVELS.length;
const MEDIAN_IDX = 2; // index of 0.50 in QUANTILE_LEVELS

// ONNX WebGPU EP: the artifact's own forceCpuNodeNames override list,
// copied inline (same as bench/onnx/index.html) rather than re-fetched.
const FORCE_CPU_NODE_NAMES = [
    'n5_3', 'n5_4', 'n5_5', 'n5_6', 'n5', 'n5_2', 'node_bitwise_not', 'node_bitwise_or', 'node_ne',
    'node_cos', 'node_sin', 'node_bitwise_not_1', 'node_bitwise_not_5', 'node_bitwise_and',
    'node_pow_2', 'node_bitwise_not_3', 'node_bitwise_not_6', 'node_pow_3', 'node_bitwise_not_8',
    'node_pow_1', 'node_bitwise_and_1', 'node_bitwise_not_7', 'node_bitwise_and_7', 'node_bitwise_and_2',
    'node_bitwise_and_8', 'node_bitwise_and_3', 'node_bitwise_and_4', 'node_bitwise_and_5',
    'node_bitwise_or_2', 'node_bitwise_and_6', 'node_sinh',
];

// onnxruntime-web is loaded lazily (CDN in Pages mode, local file in
// LOCAL_ASSETS mode) instead of a static <script> tag in index.html, so
// this same page works from either source. `ort` is a script global once
// loaded.
let ortReadyPromise = null;
function ensureOrt() {
    if (!ortReadyPromise) {
        ortReadyPromise = new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = ORT_SCRIPT_URL;
            s.onload = () => resolve();
            s.onerror = () => reject(new Error(`failed to load onnxruntime-web from ${ORT_SCRIPT_URL}`));
            document.head.appendChild(s);
        }).then(() => {
            ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
            ort.env.wasm.simd = true;
        });
    }
    return ortReadyPromise;
}

function median(a) {
    const s = [...a].sort((x, y) => x - y);
    const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function p90(a) {
    const s = [...a].sort((x, y) => x - y);
    const idx = Math.min(s.length - 1, Math.ceil(0.9 * s.length) - 1);
    return s[idx];
}

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
function setStatus(kind, text) {
    statusDot.className = 'status-dot ' + kind;
    statusText.textContent = text;
}

let state = {
    context: CH_OPTIONS[0].context,
    horizon: CH_OPTIONS[0].horizon,
    quant: 'q8_0',
    usePasted: false,
    useF32: false,
    series: null,
    lastResult: null,
};

// ---- UI scaffolding ----
function addRadioChoice(container, name, text, checked, onChange) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = name;
    input.checked = checked;
    input.addEventListener('change', onChange);
    label.appendChild(input);
    label.appendChild(document.createTextNode(text));
    container.appendChild(label);
}

const chChoices = document.getElementById('chChoices');
CH_OPTIONS.forEach((opt, i) => {
    addRadioChoice(chChoices, 'ch', `${opt.context}/${opt.horizon}`, i === 0, () => {
        state.context = opt.context;
        state.horizon = opt.horizon;
    });
});

const quantChoices = document.getElementById('quantChoices');
['q8_0', 'q4_0'].forEach((q, i) => {
    addRadioChoice(quantChoices, 'quant', q, i === 0, () => {
        state.quant = q;
    });
});

const seriesChoices = document.getElementById('seriesChoices');
const pasteArea = document.getElementById('pasteArea');
const pasteLabel = document.getElementById('pasteLabel');
['us_births (fixture)', 'paste your own'].forEach((label, i) => {
    addRadioChoice(seriesChoices, 'series', label, i === 0, () => {
        state.usePasted = i === 1;
        pasteArea.style.display = state.usePasted ? 'block' : 'none';
        pasteLabel.style.display = state.usePasted ? 'block' : 'none';
    });
});

const f32Checkbox = document.getElementById('f32Opt');
f32Checkbox.addEventListener('change', () => {
    state.useF32 = f32Checkbox.checked;
});

document.getElementById('runBtn').onclick = () => run();

// ---- data ----
async function loadFixtureSeries() {
    if (state.series) return state.series;
    const buf = await fetch(ONNX_SERIES_URL).then((r) => r.arrayBuffer());
    state.series = new Float32Array(buf);
    return state.series;
}

function contextFromPasted(context) {
    const lines = pasteArea.value.split('\n').map((s) => s.trim()).filter((s) => s.length > 0);
    const nums = lines.map(Number).filter((n) => Number.isFinite(n));
    if (nums.length < context) {
        throw new Error(`pasted series has ${nums.length} valid numbers, need at least ${context} (context)`);
    }
    return Float32Array.from(nums.slice(nums.length - context));
}

function shiftedContexts(series, context, n, origin) {
    const out = [];
    for (let r = 0; r < n; r++) {
        const end = origin - r;
        const start = end - context;
        if (start < 0) throw new Error('not enough series history for shifted batch windows');
        out.push(series.slice(start, end));
    }
    return out;
}

// ---- ONNX side ----
function buildOnnxFeeds(contextRows, context, computeHorizon) {
    const nRows = contextRows.length;
    const target_context = new Float32Array(nRows * context);
    for (let r = 0; r < nRows; r++) target_context.set(contextRows[r], r * context);
    const target_group_ids = new Int32Array(nRows);
    for (let r = 0; r < nRows; r++) target_group_ids[r] = r;
    return {
        target_context: new ort.Tensor('float32', target_context, [nRows, context]),
        target_group_ids: new ort.Tensor('int32', target_group_ids, [nRows]),
        future_covariate_context: new ort.Tensor('float32', new Float32Array(0), [0, context]),
        future_covariate_future: new ort.Tensor('float32', new Float32Array(0), [0, computeHorizon]),
        future_covariate_group_ids: new ort.Tensor('int32', new Int32Array(0), [0]),
    };
}

async function makeOnnxSession(ep) {
    await ensureOrt();
    const opts = { graphOptimizationLevel: 'basic' };
    opts.executionProviders = ep === 'webgpu' ? [{ name: 'webgpu', forceCpuNodeNames: FORCE_CPU_NODE_NAMES }, 'wasm'] : ['wasm'];
    return ort.InferenceSession.create(ONNX_MODEL_URL, opts);
}

async function loadOnnx() {
    const headResp = await fetch(ONNX_MODEL_URL, { method: 'HEAD' });
    const sizeBytes = parseInt(headResp.headers.get('content-length') || '0', 10);
    let session, ep;
    try {
        session = await makeOnnxSession('webgpu');
        ep = 'webgpu';
    } catch (e) {
        session = await makeOnnxSession('wasm');
        ep = 'wasm';
    }
    return { session, ep, sizeBytes };
}

async function onnxRun(session, contextRows, context, horizon) {
    const computeHorizon = Math.ceil(horizon / 32) * 32;
    const feeds = buildOnnxFeeds(contextRows, context, computeHorizon);
    const t0 = performance.now();
    const out = await session.run(feeds);
    const ms = performance.now() - t0;
    // out.quantiles is [nRows, computeHorizon, NQ] time-major then quantile-minor per row
    const data = out.quantiles.data;
    return { ms, data, computeHorizon };
}

// ---- ours (t0-fast) side ----
async function loadOurs(quant) {
    const t0wasm = await import(OURS_PKG);
    await t0wasm.default();
    await t0wasm.initBackend();
    const url = OURS_GGUF[quant];
    const headResp = await fetch(url, { method: 'HEAD' });
    const sizeBytes = parseInt(headResp.headers.get('content-length') || '0', 10);
    const modelBuf = await fetch(url).then((r) => r.arrayBuffer());
    const model = t0wasm.T0Wasm.load(new Uint8Array(modelBuf));
    return { model, sizeBytes, ep: 'webgpu (t0-fast)' };
}

async function oursRun(model, context, horizon) {
    const t0 = performance.now();
    const out = await model.forecast(context, horizon);
    const ms = performance.now() - t0;
    return { ms, data: out };
}

// F32-residency reference build: the raw, never-quantized model.safetensors
// (same file t0-cli export-gguf reads from) loaded through t0-fast's F32
// path (T0Wasm.loadF32) -- ground truth for the agreement table, not a
// Q8_0/Q4_0 GGUF dequantized back to f32.
async function loadOursF32() {
    const t0wasm = await import(OURS_PKG);
    await t0wasm.default();
    await t0wasm.initBackend();
    const [stBuf, configText] = await Promise.all([
        fetch(OURS_F32_SAFETENSORS).then((r) => r.arrayBuffer()),
        fetch(OURS_F32_CONFIG).then((r) => r.text()),
    ]);
    return t0wasm.T0Wasm.loadF32(new Uint8Array(stBuf), configText);
}

// ---- agreement ----
function extractMedianAndBands(data, horizon, nq, medianIdx) {
    const median_ = new Float64Array(horizon);
    const q10 = new Float64Array(horizon);
    const q90 = new Float64Array(horizon);
    for (let t = 0; t < horizon; t++) {
        median_[t] = data[t * nq + medianIdx];
        q10[t] = data[t * nq + 0];
        q90[t] = data[t * nq + nq - 1];
    }
    return { median: median_, q10, q90 };
}

function computeAgreement(theirsData, oursData, horizon, nq) {
    // range over both full quantile grids for this horizon window
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < horizon * nq; i++) {
        lo = Math.min(lo, theirsData[i], oursData[i]);
        hi = Math.max(hi, theirsData[i], oursData[i]);
    }
    const range = hi - lo || 1;

    const rows = [];
    let overallMax = 0, overallSum = 0, overallN = 0;
    for (let qi = 0; qi < nq; qi++) {
        let maxAbs = 0, sumAbs = 0;
        for (let t = 0; t < horizon; t++) {
            const d = Math.abs(theirsData[t * nq + qi] - oursData[t * nq + qi]);
            maxAbs = Math.max(maxAbs, d);
            sumAbs += d;
            overallMax = Math.max(overallMax, d);
            overallSum += d;
            overallN += 1;
        }
        rows.push({
            level: QUANTILE_LEVELS[qi],
            maxAbs,
            maxAbsPct: (maxAbs / range) * 100,
            meanAbs: sumAbs / horizon,
            meanAbsPct: (sumAbs / horizon / range) * 100,
        });
    }
    return {
        range,
        perQuantile: rows,
        overall: {
            maxAbs: overallMax,
            maxAbsPct: (overallMax / range) * 100,
            meanAbs: overallSum / overallN,
            meanAbsPct: (overallSum / overallN / range) * 100,
        },
    };
}

// ---- rendering ----
function renderLatencyTable(rows) {
    const tbody = document.querySelector('#latencyTable tbody');
    tbody.innerHTML = '';
    for (const r of rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r.engine}</td><td>${r.ep}</td><td>${r.downloadMB.toFixed(1)}</td><td>${r.cold.toFixed(1)}</td><td>${r.warmMedian.toFixed(1)}</td><td>${r.warmP90.toFixed(1)}</td>`;
        tbody.appendChild(tr);
    }
}

function renderBatchTable(rows) {
    const tbody = document.querySelector('#batchTable tbody');
    tbody.innerHTML = '';
    for (const r of rows) {
        const tr = document.createElement('tr');
        if (r.error) {
            tr.innerHTML = `<td>${r.engine}</td><td colspan="2">n/a</td><td>${r.error}</td>`;
        } else {
            tr.innerHTML = `<td>${r.engine}</td><td>${r.msTotal.toFixed(1)}</td><td>${r.msPerSignal.toFixed(2)}</td><td>${r.note || ''}</td>`;
        }
        tbody.appendChild(tr);
    }
}

// `agreement` = { oursVsF32, theirsVsF32, theirsVsOurs }, each a
// computeAgreement() result. Three rows per quantile (plus overall): the
// 1e-6-scale row (ours vs the true F32 reference) sits next to the
// quantized-vs-quantized rows so the two error sources aren't conflated.
const AGREE_NOTE = 'The 1e-6 parity is the II engine against the F32 reference with identical weights; the difference between the two quantized models is the sum of the two quantization errors.';
const AGREE_NOTE_NO_F32 = 'F32 reference rows need the "F32 reference" opt-in above (adds a 407 MB download); showing TFC (INT8) vs II (quantized) only.';
const AGREE_COMPARISONS = [
    ['theirsVsOurs', 'TFC (INT8) vs II (quantized)'],
    ['oursVsF32', 'II (quantized) vs F32 reference'],
    ['theirsVsF32', 'TFC (INT8) vs F32 reference'],
];

function renderAgreementTable(agreement) {
    const comparisons = AGREE_COMPARISONS.filter(([key]) => key in agreement);
    document.getElementById('agreeNote').textContent = comparisons.length === AGREE_COMPARISONS.length ? AGREE_NOTE : AGREE_NOTE_NO_F32;
    const tbody = document.querySelector('#agreeTable tbody');
    tbody.innerHTML = '';
    const addRow = (label, cmpLabel, r) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${label}</td><td>${cmpLabel}</td><td>${r.maxAbs.toFixed(3)}</td><td>${r.maxAbsPct.toFixed(2)}%</td><td>${r.meanAbs.toFixed(3)}</td><td>${r.meanAbsPct.toFixed(2)}%</td>`;
        tbody.appendChild(tr);
    };
    for (const [key, cmpLabel] of comparisons) addRow('overall', cmpLabel, agreement[key].overall);
    for (let qi = 0; qi < NQ; qi++) {
        const label = `Q${Math.round(QUANTILE_LEVELS[qi] * 100)}`;
        for (const [key, cmpLabel] of comparisons) addRow(label, cmpLabel, agreement[key].perQuantile[qi]);
    }
}

function drawOverlayChart(theirsBands, oursBands, horizon) {
    const canvas = document.getElementById('overlayChart');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const PAD_L = 36, PAD_R = 8, PAD_T = 8, PAD_B = 8;
    const plotW = w - PAD_L - PAD_R, plotH = h - PAD_T - PAD_B;
    let lo = Infinity, hi = -Infinity;
    for (const b of [theirsBands, oursBands]) {
        for (const arr of [b.median, b.q10, b.q90]) {
            for (const v of arr) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        }
    }
    const pad = (hi - lo) * 0.1 || 1;
    lo -= pad; hi += pad;
    const xAt = (t) => PAD_L + (t / (horizon - 1)) * plotW;
    const yAt = (v) => PAD_T + plotH - ((v - lo) / (hi - lo)) * plotH;

    ctx.strokeStyle = '#eee';
    ctx.fillStyle = '#999';
    ctx.font = '10px "IBM Plex Mono", monospace';
    for (let t = 0; t <= 4; t++) {
        const v = lo + (t / 4) * (hi - lo);
        const y = yAt(v);
        ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(w - PAD_R, y); ctx.stroke();
        ctx.fillText(Math.round(v).toLocaleString(), 2, y + 3);
    }

    function drawBand(bands, color) {
        ctx.fillStyle = color + '33';
        ctx.beginPath();
        for (let t = 0; t < horizon; t++) ctx.lineTo(xAt(t), yAt(bands.q90[t]));
        for (let t = horizon - 1; t >= 0; t--) ctx.lineTo(xAt(t), yAt(bands.q10[t]));
        ctx.closePath();
        ctx.fill();
    }
    function drawMedian(bands, color, dashed) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash(dashed ? [4, 3] : []);
        ctx.beginPath();
        for (let t = 0; t < horizon; t++) {
            const x = xAt(t), y = yAt(bands.median[t]);
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    drawBand(theirsBands, '#1a1a1a');
    drawBand(oursBands, '#2d7d46');
    drawMedian(theirsBands, '#1a1a1a', false);
    drawMedian(oursBands, '#2d7d46', true);
}


async function gpuAdapterName() {
    if (!navigator.gpu) return 'navigator.gpu unavailable (WebGPU requires HTTPS or a supporting browser)';
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return 'no adapter returned';
        if (adapter.info) {
            const i = adapter.info;
            return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' / ') || 'unknown adapter';
        }
        if (adapter.requestAdapterInfo) {
            const i = await adapter.requestAdapterInfo();
            return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' / ') || 'unknown adapter';
        }
        return 'adapter present, no info API';
    } catch (e) {
        return 'error: ' + (e.message || String(e));
    }
}

// ---- main protocol ----
async function run(overrideConfig) {
    const cfg = Object.assign({ context: state.context, horizon: state.horizon, quant: state.quant, usePasted: state.usePasted, useF32: state.useF32 }, overrideConfig || {});
    document.getElementById('runBtn').disabled = true;
    try {
        setStatus('loading', 'loading series...');
        const series = await loadFixtureSeries();
        const context = cfg.usePasted ? contextFromPasted(cfg.context) : series.slice(ORIGIN - cfg.context, ORIGIN);

        setStatus('loading', 'loading onnxruntime-web session (theirs)...');
        const theirs = await loadOnnx();

        setStatus('loading', 'loading t0-fast (' + cfg.quant + ', ours)...');
        const ours = await loadOurs(cfg.quant);

        let f32Model = null;
        if (cfg.useF32) {
            setStatus('loading', 'loading t0-fast F32 reference...');
            f32Model = await loadOursF32();
        }

        setStatus('generating', 'cold calls...');
        const theirsCold = await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
        const oursCold = await oursRun(ours.model, context, cfg.horizon);

        setStatus('generating', 'warmups...');
        for (let i = 0; i < 2; i++) {
            await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
            await oursRun(ours.model, context, cfg.horizon);
        }

        setStatus('generating', `${N_WARM} alternating calls each...`);
        const theirsTimes = [], oursTimes = [];
        let theirsLast = theirsCold, oursLast = oursCold;
        for (let i = 0; i < N_WARM; i++) {
            theirsLast = await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
            theirsTimes.push(theirsLast.ms);
            oursLast = await oursRun(ours.model, context, cfg.horizon);
            oursTimes.push(oursLast.ms);
        }

        setStatus('generating', 'batch 24...');
        const batchRows = [];
        const rows = cfg.usePasted ? Array.from({ length: N_BATCH }, () => context) : shiftedContexts(series, cfg.context, N_BATCH, ORIGIN);
        const batchNote = cfg.usePasted ? 'same window x24 (no history to shift)' : '24 distinct shifted windows (stride 1 day)';
        try {
            const { ms } = await onnxRun(theirs.session, rows, cfg.context, cfg.horizon);
            batchRows.push({ engine: 'TFC (ONNX)', msTotal: ms, msPerSignal: ms / N_BATCH, note: batchNote });
        } catch (e) {
            batchRows.push({ engine: 'TFC (ONNX)', error: e.message || String(e) });
        }
        let rowParityMaxAbs = null;
        let rowParityRange = null;
        let rowParityRelative = null;
        try {
            const concatRows = new Float32Array(N_BATCH * cfg.context);
            for (let r = 0; r < N_BATCH; r++) concatRows.set(rows[r], r * cfg.context);
            const lengths = Uint32Array.from({ length: N_BATCH }, () => cfg.context);
            const t0 = performance.now();
            const batchOut = await ours.model.forecastBatchRows(concatRows, lengths, cfg.horizon);
            const ms = performance.now() - t0;
            batchRows.push({ engine: 'II (t0-fast)', msTotal: ms, msPerSignal: ms / N_BATCH, note: batchNote });

            // Headless parity gate: every row of forecastBatchRows must
            // equal that row's own single-signal forecast() to within a
            // *relative* tolerance -- max-abs diff / (max-min) of this
            // window's own full quantile-grid range, not an absolute
            // epsilon. An absolute 1e-4 gate was previously calibrated
            // against a small-magnitude synthetic fixture (values in
            // [-1, 1]) and falsely failed real-world thousands-magnitude
            // data (see docs/runs/2026-09-20-compare-page.md): the same
            // 0.001953125 absolute diff is ~1e-6 relative there.
            setStatus('generating', 'batch-rows parity check...');
            const perRow = cfg.horizon * NQ;
            let maxAbs = 0;
            let rpLo = Infinity, rpHi = -Infinity;
            for (let r = 0; r < N_BATCH; r++) {
                const single = await ours.model.forecast(rows[r], cfg.horizon);
                for (let i = 0; i < perRow; i++) {
                    const bo = batchOut[r * perRow + i];
                    maxAbs = Math.max(maxAbs, Math.abs(bo - single[i]));
                    rpLo = Math.min(rpLo, bo, single[i]);
                    rpHi = Math.max(rpHi, bo, single[i]);
                }
            }
            rowParityRange = rpHi - rpLo || 1;
            rowParityRelative = maxAbs / rowParityRange;
            rowParityMaxAbs = maxAbs;
            if (rowParityRelative > 1e-5) {
                throw new Error(
                    `forecastBatchRows parity gate failed: max-abs diff ${maxAbs} / range ${rowParityRange} = ${rowParityRelative} > 1e-5 across ${N_BATCH} rows`
                );
            }
        } catch (e) {
            batchRows.push({ engine: 'II (t0-fast)', error: e.message || String(e) });
        }

        // agreement: computeHorizon on theirs side may exceed cfg.horizon
        // (padded to a multiple of 32) -- slice both down to cfg.horizon.
        const theirsFull = theirsLast.data;
        const oursFull = oursLast.data;
        const theirsSliced = new Float64Array(cfg.horizon * NQ);
        for (let t = 0; t < cfg.horizon; t++) for (let qi = 0; qi < NQ; qi++) theirsSliced[t * NQ + qi] = theirsFull[t * NQ + qi];
        const agreement = {
            theirsVsOurs: computeAgreement(theirsSliced, oursFull, cfg.horizon, NQ),
        };
        if (cfg.useF32) {
            setStatus('generating', 'F32 reference forecast...');
            const f32Full = await oursRun(f32Model, context, cfg.horizon).then((r) => r.data);
            agreement.oursVsF32 = computeAgreement(oursFull, f32Full, cfg.horizon, NQ);
            agreement.theirsVsF32 = computeAgreement(theirsSliced, f32Full, cfg.horizon, NQ);
        }
        const theirsBands = extractMedianAndBands(theirsSliced, cfg.horizon, NQ, MEDIAN_IDX);
        const oursBands = extractMedianAndBands(oursFull, cfg.horizon, NQ, MEDIAN_IDX);

        const latencyRows = [
            {
                engine: 'TFC (ONNX INT8)', ep: theirs.ep,
                downloadMB: theirs.sizeBytes / 1e6,
                cold: theirsCold.ms, warmMedian: median(theirsTimes), warmP90: p90(theirsTimes),
            },
            {
                engine: `II (t0-fast ${cfg.quant})`, ep: ours.ep,
                downloadMB: ours.sizeBytes / 1e6,
                cold: oursCold.ms, warmMedian: median(oursTimes), warmP90: p90(oursTimes),
            },
        ];

        renderLatencyTable(latencyRows);
        renderBatchTable(batchRows);
        renderAgreementTable(agreement);
        drawOverlayChart(theirsBands, oursBands, cfg.horizon);

        const gpuInfo = await gpuAdapterName();

        setStatus('ready', `done: context=${cfg.context} horizon=${cfg.horizon} quant=${cfg.quant}`);

        const result = { config: cfg, latencyRows, batchRows, agreement, rowParityMaxAbs, rowParityRange, rowParityRelative, gpuInfo, ua: navigator.userAgent, timestamp: new Date().toISOString() };
        state.lastResult = result;
        return result;
    } catch (e) {
        const msg = e.message || String(e);
        // A stale pkg-fast bundle (built before a wasm-bindgen export was
        // added, e.g. forecastBatchRows/loadF32) throws a raw JS
        // "X is not a function" TypeError -- surface that as an actionable
        // status instead of the confusing raw message.
        if (e instanceof TypeError && /is not a function/.test(msg)) {
            setStatus('loading', 'rebuilding engine...');
        } else {
            setStatus('error', 'error: ' + msg);
        }
        throw e;
    } finally {
        document.getElementById('runBtn').disabled = false;
    }
}

setStatus('ready', 'ready, choose context, horizon and quant, then run');

window.__app = {
    run,
    getState: () => state,
};
