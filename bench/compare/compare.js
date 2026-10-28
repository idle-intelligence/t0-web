// bench/compare/compare.js
//
// Single-page head-to-head: theforecastingcompany's t0-alpha-onnx-int8
// export (onnxruntime-web, WebGPU EP with wasm fallback) vs our GGUF-
// resident t0-fast (Q8_0/Q4_0, crates/t0-wasm/pkg-fast). Both engines load
// in this same tab so the numbers are for whatever machine opens this page.
//
// Driven either by the UI (run button) or headlessly via window.__app.run().

// The onnxruntime-web version the official INT8 export's manifest.json
// names.
const ORT_VERSION = '1.29.0';
const ORT_CDN_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_SCRIPT_URL = `${ORT_CDN_BASE}ort.webgpu.min.js`;
const ORT_WASM_PATHS = ORT_CDN_BASE;

// Same engine-build version tag web/worker.js uses on its wasm URLs, so a
// hard reload here picks up a rebuilt pkg-wgpu the same way the demo does.
const ENGINE_BUILD = '2026-10-28.2';

// The Forecasting Company's export, pinned to one revision so the page and
// the node list below always describe the same graph.
const ONNX_REVISION = '95c0ad00fbb08d1bd5f72d3e6274f6f2e80f45c0';
const ONNX_MODEL_URL = `https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8/resolve/${ONNX_REVISION}/t0-alpha-grouped-int8.onnx`;
const ONNX_SERIES_URL = '../../web/data/series.f32';
const OURS_PKG = `../../web/pkg-wgpu/t0_wasm.js?v=${ENGINE_BUILD}`;
const OURS_WASM = `../../web/pkg-wgpu/t0_wasm_bg.wasm?v=${ENGINE_BUILD}`;
const OURS_GGUF = {
    q8_0: 'https://huggingface.co/idle-intelligence/t0-alpha-q8_0-webgpu/resolve/main/t0-alpha-q8_0.gguf',
    q4_0: 'https://huggingface.co/idle-intelligence/t0-alpha-q4_0-webgpu/resolve/main/t0-alpha-q4_0.gguf',
};
// F32 reference is opt-in (407 MB download) -- see the checkbox wiring below.
const OURS_F32_SAFETENSORS = 'https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/model.safetensors';
const OURS_F32_CONFIG = 'https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/config.json';

const CH_OPTIONS = [
    { context: 512, horizon: 32 },
    { context: 512, horizon: 128 },
    { context: 256, horizon: 64 },
    { context: 2048, horizon: 128 },
];
const ORIGIN = 3600; // same fixture window as the head-to-head benchmark
const N_BATCH = 24;
const N_WARM = 10;
const IDLE_MS = 2000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90];
const NQ = QUANTILE_LEVELS.length;
const MEDIAN_IDX = 2; // index of 0.50 in QUANTILE_LEVELS

// ONNX WebGPU EP: the forceCpuNodeNames list from the export's own
// webgpu-options.js at revision 95c0ad00fbb08d1bd5f72d3e6274f6f2e80f45c0 (the same revision as
// ONNX_MODEL_URL), copied verbatim. Its graphOptimizationLevel, 'basic', is
// the one makeOnnxSession uses.
const FORCE_CPU_NODE_NAMES = [
    'node_Shape_5', 'node_Shape_14', 'node_Shape_49', 'node_Shape_0', 'node_Shape_1',
    'node_Shape_47', 'node_sym_size_int_90', 'node_sym_size_int_6', 'node_Concat_80', 'n5_3',
    'n5_4', 'n5_5', 'n5_6', 'node_sym_size_int_85', 'node_Concat_42', 'node_Concat_4208',
    'node_sym_size_int_86', 'n5', 'n5_2', 'node__to_copy_4', 'node_floordiv_1', 'node_add_9003',
    'node_mul_7525', 'node_add_5', 'node_view', 'node_Reshape_131', 'node_Shape_45', 'node_mod_1',
    'node_expand_2', 'node__to_copy_5', 'node_add_6', 'node_Reshape_20', 'node_add_95',
    'node_view_1', 'node_floordiv', 'node_Reshape_132', 'node_Concat_23', 'node_Reshape_56',
    'node_arange', 'node_floordiv_3', 'node_expand_3', 'node_Reshape_147', 'node_arange_1',
    'node_Concat_133', 'node_Concat_57', 'node_sub_3869', 'node_Concat_149', 'node_Concat_170',
    'node_Concat_227', 'node_Concat_308', 'node_Concat_326', 'node_Concat_419', 'node_Concat_647',
    'node_Concat_701', 'node_Concat_4140', 'node_unsqueeze_6', 'node_unsqueeze_7', 'node_expand',
    'node_expand_1', 'node_add_8987', 'node_Reshape_4196', 'node_Shape_70', 'node_Shape_72',
    'node_Reshape_4199', 'node_Shape_91', 'node_ones_like', 'node_ones_like_1', 'node_ne',
    'node_bitwise_or_2', 'node_neg', 'node_neg_1', 'node_bitwise_or', 'node_bitwise_or_3',
    'node_ne_5', 'node_where_9', 'node_where_10', 'node_cos', 'node_sin', 'node_bitwise_not',
    'node_Cast_180', 'node_cat_8', 'node_cat_9', 'node_argmax', 'node_ReduceMax_183', 'node_view_7',
    'node_view_6', 'node_gather_1', 'node_bitwise_and_1', 'node_bitwise_or_1', 'node_gather',
    'node_pow_1', 'node_squeeze_1', 'node_bitwise_and_2', 'node_squeeze', 'node_bitwise_and',
    'node_where_22', 'node_where_21', 'node_unsqueeze_2', 'node_unsqueeze_3', 'node_permute',
    'node_clamp_6', 'node_unsqueeze_10', 'node_unsqueeze_11', 'node_pow_2', 'node_bitwise_and_3',
    'node_bitwise_not_3', 'node_bitwise_and_4', 'node_bitwise_and_9', 'node_bitwise_not_4',
    'node_bitwise_and_5', 'node_bitwise_and_6', 'node_bitwise_and_10', 'node_bitwise_and_11',
    'node_bitwise_and_7', 'node_bitwise_not_5', 'node_bitwise_or_4', 'node_bitwise_and_12',
    'node_Shape_105', 'node_bitwise_and_8', 'node_Shape_144', 'node_Shape_390', 'node_Slice_392',
    'node_Slice_394', 'node_Slice_396', 'node_Concat_398', 'node_Concat_401', 'node_Shape_599',
    'node_Slice_601', 'node_Slice_602', 'node_Slice_604', 'node_Concat_606', 'node_Concat_609',
    'node_Shape_673', 'node_Slice_675', 'node_Slice_676', 'node_Slice_678', 'node_Concat_680',
    'node_Concat_683', 'node_Shape_880', 'node_Slice_882', 'node_Slice_883', 'node_Slice_885',
    'node_Concat_887', 'node_Concat_890', 'node_Shape_1087', 'node_Slice_1089', 'node_Slice_1090',
    'node_Slice_1092', 'node_Concat_1094', 'node_Concat_1097', 'node_Shape_1161', 'node_Slice_1163',
    'node_Slice_1164', 'node_Slice_1166', 'node_Concat_1168', 'node_Concat_1171', 'node_Shape_1368',
    'node_Slice_1370', 'node_Slice_1371', 'node_Slice_1373', 'node_Concat_1375', 'node_Concat_1378',
    'node_Shape_1575', 'node_Slice_1577', 'node_Slice_1578', 'node_Slice_1580', 'node_Concat_1582',
    'node_Concat_1585', 'node_Shape_1649', 'node_Slice_1651', 'node_Slice_1652', 'node_Slice_1654',
    'node_Concat_1656', 'node_Concat_1659', 'node_Shape_1856', 'node_Slice_1858', 'node_Slice_1859',
    'node_Slice_1861', 'node_Concat_1863', 'node_Concat_1866', 'node_Shape_2063', 'node_Slice_2065',
    'node_Slice_2066', 'node_Slice_2068', 'node_Concat_2070', 'node_Concat_2073', 'node_Shape_2137',
    'node_Slice_2139', 'node_Slice_2140', 'node_Slice_2142', 'node_Concat_2144', 'node_Concat_2147',
    'node_Shape_2344', 'node_Slice_2346', 'node_Slice_2347', 'node_Slice_2349', 'node_Concat_2351',
    'node_Concat_2354', 'node_Shape_2551', 'node_Slice_2553', 'node_Slice_2554', 'node_Slice_2556',
    'node_Concat_2558', 'node_Concat_2561', 'node_Shape_2625', 'node_Slice_2627', 'node_Slice_2628',
    'node_Slice_2630', 'node_Concat_2632', 'node_Concat_2635', 'node_Shape_2832', 'node_Slice_2834',
    'node_Slice_2835', 'node_Slice_2837', 'node_Concat_2839', 'node_Concat_2842', 'node_Shape_3039',
    'node_Slice_3041', 'node_Slice_3042', 'node_Slice_3044', 'node_Concat_3046', 'node_Concat_3049',
    'node_Shape_3113', 'node_Slice_3115', 'node_Slice_3116', 'node_Slice_3118', 'node_Concat_3120',
    'node_Concat_3123', 'node_Shape_3320', 'node_Slice_3322', 'node_Slice_3323', 'node_Slice_3325',
    'node_Concat_3327', 'node_Concat_3330', 'node_Shape_3527', 'node_Slice_3529', 'node_Slice_3530',
    'node_Slice_3532', 'node_Concat_3534', 'node_Concat_3537', 'node_Shape_3601', 'node_Slice_3603',
    'node_Slice_3604', 'node_Slice_3606', 'node_Concat_3608', 'node_Concat_3611', 'node_Shape_3808',
    'node_Slice_3810', 'node_Slice_3811', 'node_Slice_3813', 'node_Concat_3815', 'node_Concat_3818',
    'node_Shape_4015', 'node_Slice_4017', 'node_Slice_4018', 'node_Slice_4020', 'node_Concat_4022',
    'node_Concat_4025', 'node_Shape_4089', 'node_Slice_4091', 'node_Slice_4092', 'node_Slice_4094',
    'node_Concat_4096', 'node_Concat_4099', 'node_sinh',
];

// onnxruntime-web is loaded lazily from the CDN, only when a run starts.
// `ort` is a script global once loaded.
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
    await t0wasm.default({ module_or_path: new URL(OURS_WASM, import.meta.url).href });
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
    await t0wasm.default({ module_or_path: new URL(OURS_WASM, import.meta.url).href });
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
        tr.innerHTML = `<td>${r.engine}</td><td>${r.ep}</td><td>${r.downloadMB.toFixed(1)}</td><td>${r.setup.toFixed(1)}</td><td>${r.cold.toFixed(1)}</td><td>${r.warmMedian.toFixed(1)}</td><td>${r.warmP90.toFixed(1)}</td>`;
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
const AGREE_NOTE = 'The 1e-6 parity is t0-fast against the F32 reference with identical weights; the difference between the two quantized models is the sum of the two quantization errors.';
const AGREE_NOTE_NO_F32 = 'F32 reference rows need the "F32 reference" opt-in above (adds a 407 MB download); showing the t0 ONNX INT8 export vs t0-fast only.';
const AGREE_COMPARISONS = [
    ['theirsVsOurs', 't0 ONNX INT8 export vs t0-fast'],
    ['oursVsF32', 't0-fast vs F32 reference'],
    ['theirsVsF32', 't0 ONNX INT8 export vs F32 reference'],
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

        setStatus('loading', 'loading the t0 ONNX INT8 export...');
        const theirs = await loadOnnx();

        setStatus('loading', 'loading t0-fast (' + cfg.quant + ')...');
        const ours = await loadOurs(cfg.quant);

        let f32Model = null;
        if (cfg.useF32) {
            setStatus('loading', 'loading the F32 reference (407 MB)...');
            f32Model = await loadOursF32();
        }

        // One untimed forecast per engine absorbs pipeline and kernel setup,
        // which is a property of the runtime, not of the model.
        setStatus('generating', 'first forecast on each engine, with setup...');
        const theirsSetup = await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
        const oursSetup = await oursRun(ours.model, context, cfg.horizon);

        // Isolated call: one forecast after the GPU has sat idle for two
        // seconds, measured the same way on both engines.
        setStatus('generating', 'isolated forecast on each engine...');
        await sleep(IDLE_MS);
        const theirsCold = await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
        await sleep(IDLE_MS);
        const oursCold = await oursRun(ours.model, context, cfg.horizon);

        setStatus('generating', 'warming up both engines...');
        for (let i = 0; i < 2; i++) {
            await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
            await oursRun(ours.model, context, cfg.horizon);
        }

        setStatus('generating', `timing ${N_WARM} forecasts per engine, alternating...`);
        const theirsTimes = [], oursTimes = [];
        let theirsLast = theirsCold, oursLast = oursCold;
        for (let i = 0; i < N_WARM; i++) {
            theirsLast = await onnxRun(theirs.session, [context], cfg.context, cfg.horizon);
            theirsTimes.push(theirsLast.ms);
            oursLast = await oursRun(ours.model, context, cfg.horizon);
            oursTimes.push(oursLast.ms);
        }

        setStatus('generating', 'batch of 24 signals...');
        const batchRows = [];
        const rows = cfg.usePasted ? Array.from({ length: N_BATCH }, () => context) : shiftedContexts(series, cfg.context, N_BATCH, ORIGIN);
        const batchNote = cfg.usePasted ? 'same window x24 (no history to shift)' : '24 distinct shifted windows (stride 1 day)';
        try {
            const { ms } = await onnxRun(theirs.session, rows, cfg.context, cfg.horizon);
            batchRows.push({ engine: 't0 ONNX INT8 export', msTotal: ms, msPerSignal: ms / N_BATCH, note: batchNote });
        } catch (e) {
            batchRows.push({ engine: 't0 ONNX INT8 export', error: e.message || String(e) });
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
            batchRows.push({ engine: 't0-fast', msTotal: ms, msPerSignal: ms / N_BATCH, note: batchNote });

            // Headless parity gate: every row of forecastBatchRows must
            // equal that row's own single-signal forecast() to within a
            // *relative* tolerance -- max-abs diff / (max-min) of this
            // window's own full quantile-grid range, not an absolute
            // epsilon: on data in the thousands, float32 rounding alone
            // gives absolute diffs around 1e-3 (~1e-6 relative).
            setStatus('generating', 'checking the batch path against single forecasts...');
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
            batchRows.push({ engine: 't0-fast', error: e.message || String(e) });
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
                engine: 't0 ONNX INT8 export', ep: theirs.ep,
                downloadMB: theirs.sizeBytes / 1e6,
                setup: theirsSetup.ms, cold: theirsCold.ms, warmMedian: median(theirsTimes), warmP90: p90(theirsTimes),
            },
            {
                engine: `t0-fast ${cfg.quant}`, ep: ours.ep,
                downloadMB: ours.sizeBytes / 1e6,
                setup: oursSetup.ms, cold: oursCold.ms, warmMedian: median(oursTimes), warmP90: p90(oursTimes),
            },
        ];

        renderLatencyTable(latencyRows);
        renderBatchTable(batchRows);
        renderAgreementTable(agreement);
        drawOverlayChart(theirsBands, oursBands, cfg.horizon);

        const gpuInfo = await gpuAdapterName();

        setStatus('ready', `done, context ${cfg.context}, horizon ${cfg.horizon}, ${cfg.quant}`);

        const result = { config: cfg, latencyRows, batchRows, agreement, rowParityMaxAbs, rowParityRange, rowParityRelative, gpuInfo, ua: navigator.userAgent, timestamp: new Date().toISOString() };
        state.lastResult = result;
        return result;
    } catch (e) {
        const msg = e.message || String(e);
        setStatus('error', 'error: ' + msg);
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
