// Headless runner for bench/onnx/index.html: drives the official
// t0-alpha-onnx-int8 export through onnxruntime-web in Playwright's bundled
// Chromium (never the maintainer's browser). Reports cold/warm/batch timing and
// the raw first-forecast output (for the F32-reference verification diff,
// computed by scripts/headless/compare_onnx_ref.mjs).
//
// Usage: node scripts/headless/run_onnx_bench.mjs [--url http://127.0.0.1:PORT/]
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
const url = args.url || 'http://127.0.0.1:8032/';
const EXECUTABLE_PATH =
  '~/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
const page = await browser.newPage();
page.on('console', (msg) => console.log('[page]', msg.text()));
page.on('pageerror', (err) => console.error('[pageerror]', err.message));

console.log('Loading', url);
await page.goto(url, { waitUntil: 'load' });
console.log('running bench (this loads a 107MB model, may take a while)...');
const results = await page.evaluate(() => window.__bench.run());
console.log(JSON.stringify(results, null, 2));

await browser.close();
