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

const ONNX_MODEL_URL = '../onnx/t0-alpha-grouped-int8.onnx';
const ONNX_SERIES_URL = '../onnx/series.f32';
const OURS_PKG = '../ours/pkg-fast/t0_wasm.js';
const OURS_GGUF = { q8_0: '../ours/t0-alpha-q8_0.gguf', q4_0: '../ours/t0-alpha-q4_0.gguf' };

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

ort.env.wasm.wasmPaths = '../onnx/';
ort.env.wasm.simd = true;

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
    series: null,
    lastResult: null,
};

// ---- UI scaffolding ----
const chChoices = document.getElementById('chChoices');
CH_OPTIONS.forEach((opt, i) => {
    const b = document.createElement('button');
    b.className = 'choice-btn' + (i === 0 ? ' active' : '');
    b.textContent = `${opt.context}/${opt.horizon}`;
    b.onclick = () => {
        [...chChoices.children].forEach((c) => c.classList.remove('active'));
        b.classList.add('active');
        state.context = opt.context;
        state.horizon = opt.horizon;
    };
    chChoices.appendChild(b);
});

const quantChoices = document.getElementById('quantChoices');
['q8_0', 'q4_0'].forEach((q, i) => {
    const b = document.createElement('button');
    b.className = 'choice-btn' + (i === 0 ? ' active' : '');
    b.textContent = q;
    b.onclick = () => {
        [...quantChoices.children].forEach((c) => c.classList.remove('active'));
        b.classList.add('active');
        state.quant = q;
    };
    quantChoices.appendChild(b);
});

const seriesChoices = document.getElementById('seriesChoices');
const pasteArea = document.getElementById('pasteArea');
const pasteLabel = document.getElementById('pasteLabel');
['us_births (fixture)', 'paste your own'].forEach((label, i) => {
    const b = document.createElement('button');
    b.className = 'choice-btn' + (i === 0 ? ' active' : '');
    b.textContent = label;
    b.onclick = () => {
        [...seriesChoices.children].forEach((c) => c.classList.remove('active'));
        b.classList.add('active');
        state.usePasted = i === 1;
        pasteArea.style.display = state.usePasted ? 'block' : 'none';
        pasteLabel.style.display = state.usePasted ? 'block' : 'none';
    };
    seriesChoices.appendChild(b);
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

function renderAgreementTable(agreement) {
    const tbody = document.querySelector('#agreeTable tbody');
    tbody.innerHTML = '';
    const overallRow = document.createElement('tr');
    overallRow.innerHTML = `<td><b>overall</b></td><td>${agreement.overall.maxAbs.toFixed(3)}</td><td>${agreement.overall.maxAbsPct.toFixed(2)}%</td><td>${agreement.overall.meanAbs.toFixed(3)}</td><td>${agreement.overall.meanAbsPct.toFixed(2)}%</td>`;
    tbody.appendChild(overallRow);
    for (const r of agreement.perQuantile) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>Q${Math.round(r.level * 100)}</td><td>${r.maxAbs.toFixed(3)}</td><td>${r.maxAbsPct.toFixed(2)}%</td><td>${r.meanAbs.toFixed(3)}</td><td>${r.meanAbsPct.toFixed(2)}%</td>`;
        tbody.appendChild(tr);
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

function renderPerformance(gpuInfo) {
    document.getElementById('perfUA').textContent = navigator.userAgent;
    document.getElementById('perfGPU').textContent = 'GPU adapter: ' + gpuInfo;
    document.getElementById('perfTime').textContent = new Date().toISOString();
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
    const cfg = Object.assign({ context: state.context, horizon: state.horizon, quant: state.quant, usePasted: state.usePasted }, overrideConfig || {});
    document.getElementById('runBtn').disabled = true;
    try {
        setStatus('loading', 'loading series...');
        const series = await loadFixtureSeries();
        const context = cfg.usePasted ? contextFromPasted(cfg.context) : series.slice(ORIGIN - cfg.context, ORIGIN);

        setStatus('loading', 'loading onnxruntime-web session (theirs)...');
        const theirs = await loadOnnx();

        setStatus('loading', 'loading t0-fast (' + cfg.quant + ', ours)...');
        const ours = await loadOurs(cfg.quant);

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
        try {
            const rows = cfg.usePasted ? Array.from({ length: N_BATCH }, () => context) : shiftedContexts(series, cfg.context, N_BATCH, ORIGIN);
            const { ms } = await onnxRun(theirs.session, rows, cfg.context, cfg.horizon);
            batchRows.push({ engine: 'theirs (ONNX)', msTotal: ms, msPerSignal: ms / N_BATCH, note: cfg.usePasted ? 'same window x24 (no history to shift)' : '24 shifted windows (stride 1)' });
        } catch (e) {
            batchRows.push({ engine: 'theirs (ONNX)', error: e.message || String(e) });
        }
        try {
            const t0 = performance.now();
            await ours.model.forecastBatch(context, N_BATCH, cfg.horizon, N_BATCH);
            const ms = performance.now() - t0;
            batchRows.push({ engine: 'ours (t0-fast)', msTotal: ms, msPerSignal: ms / N_BATCH, note: 'same window x24 (forecastBatch takes one context, API constraint)' });
        } catch (e) {
            batchRows.push({ engine: 'ours (t0-fast)', error: e.message || String(e) });
        }

        // agreement: computeHorizon on theirs side may exceed cfg.horizon
        // (padded to a multiple of 32) -- slice both down to cfg.horizon.
        const theirsFull = theirsLast.data;
        const oursFull = oursLast.data;
        const theirsSliced = new Float64Array(cfg.horizon * NQ);
        for (let t = 0; t < cfg.horizon; t++) for (let qi = 0; qi < NQ; qi++) theirsSliced[t * NQ + qi] = theirsFull[t * NQ + qi];
        const agreement = computeAgreement(theirsSliced, oursFull, cfg.horizon, NQ);
        const theirsBands = extractMedianAndBands(theirsSliced, cfg.horizon, NQ, MEDIAN_IDX);
        const oursBands = extractMedianAndBands(oursFull, cfg.horizon, NQ, MEDIAN_IDX);

        const latencyRows = [
            {
                engine: 'theirs (ONNX INT8)', ep: theirs.ep,
                downloadMB: theirs.sizeBytes / 1e6,
                cold: theirsCold.ms, warmMedian: median(theirsTimes), warmP90: p90(theirsTimes),
            },
            {
                engine: `ours (t0-fast ${cfg.quant})`, ep: ours.ep,
                downloadMB: ours.sizeBytes / 1e6,
                cold: oursCold.ms, warmMedian: median(oursTimes), warmP90: p90(oursTimes),
            },
        ];

        renderLatencyTable(latencyRows);
        renderBatchTable(batchRows);
        renderAgreementTable(agreement);
        drawOverlayChart(theirsBands, oursBands, cfg.horizon);

        const gpuInfo = await gpuAdapterName();
        renderPerformance(gpuInfo);

        setStatus('ready', `done: context=${cfg.context} horizon=${cfg.horizon} quant=${cfg.quant}`);

        const result = { config: cfg, latencyRows, batchRows, agreement, gpuInfo, ua: navigator.userAgent, timestamp: new Date().toISOString() };
        state.lastResult = result;
        return result;
    } catch (e) {
        setStatus('error', 'error: ' + (e.message || String(e)));
        throw e;
    } finally {
        document.getElementById('runBtn').disabled = false;
    }
}

setStatus('ready', 'ready — choose context/horizon and quant, then run');

window.__app = {
    run,
    getState: () => state,
};
