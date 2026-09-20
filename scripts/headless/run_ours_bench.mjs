// Headless runner for bench/ours/index.html: t0-wasm (our port) through the
// same protocol as scripts/headless/run_onnx_bench.mjs, same Chromium
// build, same fixture/origin -- cold/warm-median-of-10/24-signal batch,
// per backend (webgpu / ndarray-wasm) and quant (Q8_0 / Q4_0).
//
// Usage: node scripts/headless/run_ours_bench.mjs --url http://127.0.0.1:PORT/ --pkg ./pkg-wgpu --gguf ./t0-alpha-q8_0.gguf --label "Q8_0 WebGPU"
import { chromium } from 'playwright';

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) out[key] = true;
    else { out[key] = next; i++; }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const url = args.url || 'http://127.0.0.1:8033/';
const pkg = args.pkg || './pkg-wgpu';
const gguf = args.gguf || './t0-alpha-q8_0.gguf';
const label = args.label || `${pkg}/${gguf}`;
const EXECUTABLE_PATH =
  '~/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
const page = await browser.newPage();
page.on('console', (msg) => console.log('[page]', msg.text()));
page.on('pageerror', (err) => console.error('[pageerror]', err.message));

console.log('Loading', url, 'label:', label);
await page.goto(url, { waitUntil: 'load' });
const results = await page.evaluate(([pkg, gguf, label]) => window.__bench.benchOne(pkg, gguf, label), [pkg, gguf, label]);
console.log(JSON.stringify(results, null, 2));

await browser.close();
