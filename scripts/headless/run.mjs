// Headless smoke test for web/index.html: loads the page in Playwright's
// bundled Chromium (never the maintainer's real browser -- CLAUDE.md).
//
// No origin control of any kind on this page (removed per owner override):
// the data (US births, July-November 1988) and the origin (December 1,
// 1988) are fixed. The model forecasts once, automatically on load (also
// re-runnable via the single "Forecast" button). This script covers:
//   1. page loads, the automatic forecast runs exactly once.
//   2. the forecast is 32 points x 5 quantiles, all finite.
//   3. MAE is computed against the real December 1988 values and is finite.
//   4. no range/color/native slider inputs anywhere in the DOM.
//
// Backend is auto-selected by web/worker.js (WebGPU if navigator.gpu
// exists, else WASM CPU/burn-ndarray) -- this bundled Chromium-for-Testing
// exposes navigator.gpu with a real hardware adapter on http origins with
// no extra launch flags needed (confirmed this session on this Mac; see
// docs/runs/2026-09-19-web-smoke.md).
//
// Usage: node scripts/headless/run.mjs [--url http://127.0.0.1:PORT/]
import { fileURLToPath } from 'node:url';
import path from 'node:path';

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
const URL_ = args.url ?? 'http://127.0.0.1:8031/';
const TIMEOUT_LOAD = parseInt(args['timeout-load'] ?? String(5 * 60 * 1000), 10);

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const { chromium } = await import('playwright');

// Same cached Chromium-for-Testing binary this session verified works
// standalone (`node -e "require('playwright').chromium.launch(...)"`) --
// passed explicitly so this doesn't depend on playwright's own browser
// revision bookkeeping matching whatever happens to be cached.
const EXECUTABLE_PATH =
  '~/Library/Caches/ms-playwright/chromium-1229/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

async function main() {
  const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
  const page = await browser.newPage();

  const consoleLines = [];
  page.on('console', (msg) => consoleLines.push(msg.text()));
  page.on('pageerror', (err) => consoleLines.push(`[pageerror] ${err.message}`));

  const report = { url: URL_, pass: false };

  try {
    console.log(`Loading ${URL_}`);
    const t0 = Date.now();
    await page.goto(URL_, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__app && window.__app.ready === true, undefined, { timeout: TIMEOUT_LOAD });
    const loadMs = Date.now() - t0;
    console.log(`Ready in ${loadMs} ms`);
    report.loadMs = loadMs;

    // The forecast runs automatically once the model is loaded.
    await page.waitForFunction(() => window.__app.lastForecast !== null, undefined, { timeout: 30000 });

    const backend = await page.evaluate(() => document.getElementById('metricBackend').textContent);
    console.log(`Backend: ${backend}`);
    report.backend = backend;

    const forecastCount = await page.evaluate(() => window.__app.forecastCount);
    console.log(`forecastCount after load: ${forecastCount}`);
    if (forecastCount !== 1) {
      throw new Error(`expected exactly 1 automatic forecast on load, got ${forecastCount}`);
    }

    const result = await page.evaluate(() => window.__app.lastForecast);
    const { quantiles, nQuantiles, horizon, ms } = result;
    const lengthOk = horizon === 32 && nQuantiles === 5 && quantiles.length === horizon * nQuantiles;
    const allFinite = quantiles.every((v) => Number.isFinite(v));
    console.log(`forecast: horizon=${horizon} nQuantiles=${nQuantiles} length_ok=${lengthOk} finite=${allFinite} ms=${ms.toFixed(1)}`);
    if (!lengthOk || !allFinite) {
      throw new Error(`forecast shape/finite gate failed: ${JSON.stringify({ lengthOk, allFinite, horizon, nQuantiles, len: quantiles.length })}`);
    }

    const { mae, smape } = await page.evaluate(() => ({
      mae: document.getElementById('metricMae').textContent,
      smape: document.getElementById('metricSmape').textContent,
    }));
    const metricsFinite = Number.isFinite(parseFloat(mae)) && Number.isFinite(parseFloat(smape));
    console.log(`MAE=${mae} sMAPE=${smape}% (against real December 1988 values) finite=${metricsFinite}`);
    if (!metricsFinite) {
      throw new Error(`MAE/sMAPE not finite: MAE=${mae} sMAPE=${smape}`);
    }

    // --- no range/native slider inputs anywhere on the page ---
    const inputCounts = await page.evaluate(() => ({
      total: document.querySelectorAll('input').length,
      range: document.querySelectorAll('input[type="range"]').length,
      color: document.querySelectorAll('input[type="color"]').length,
    }));
    console.log(`input elements: ${JSON.stringify(inputCounts)}`);
    if (inputCounts.total !== 0) {
      throw new Error(`page has ${inputCounts.total} <input> element(s), expected 0 (no origin control of any kind)`);
    }

    // --- STATUS text passes through the expected states ---
    const statusText = await page.evaluate(() => document.getElementById('status-text').textContent);
    console.log(`final STATUS text: "${statusText}"`);
    if (!/^forecast in \d+ ms$/.test(statusText)) {
      throw new Error(`STATUS text after the automatic forecast should be "forecast in <ms> ms", got "${statusText}"`);
    }

    report.pass = true;
    console.log(`\nPASS -- forecast ${ms.toFixed(1)} ms, MAE ${mae}, sMAPE ${smape}% (backend: ${backend})`);
  } catch (err) {
    report.error = String(err && err.message ? err.message : err);
    console.log('FAIL:', report.error);
    console.log('--- console log ---');
    console.log(consoleLines.join('\n'));
  }

  await browser.close();
  process.exitCode = report.pass ? 0 : 1;
}

main().catch((err) => {
  console.error('run.mjs fatal error:', err);
  process.exitCode = 1;
});
