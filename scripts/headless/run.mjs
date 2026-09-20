// Headless smoke test for web/index.html: loads the page in Playwright's
// bundled Chromium (never a personal browser -- CLAUDE.md).
//
// The plotted history (US births, July-November 1988) never changes; only
// the origin (where the forecast starts) is draggable within it, drawn on
// the INPUT canvas as a 1px dashed line + hollow circle (no native
// control). This script covers:
//   1. page loads, the automatic forecast runs exactly once.
//   2. the forecast is 32 points x 5 quantiles, all finite.
//   3. inside-band / CRPS / vs-weekly-naive are finite and sane (inside
//      band in [0, horizon], naive ratio > 0).
//   4. a real mouse drag on the INPUT chart produces exactly one forecast
//      (debounced, commit-on-release -- not one per pointermove).
//   5. dragging never changes the drawn data (checksum of the July-
//      November window before/after) and the origin is clamped within
//      window.__app.originBounds().
//   6. no range/color/native slider inputs anywhere in the DOM.
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
import os from 'node:os';

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
  process.env.CHROMIUM_PATH ||
  path.join(os.homedir(), 'Library/Caches/ms-playwright/chromium-1229/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing');

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

    const { inside, crps, naive } = await page.evaluate(() => ({
      inside: document.getElementById('metricInside').textContent,
      crps: document.getElementById('metricCrps').textContent,
      naive: document.getElementById('metricNaive').textContent,
    }));
    const insideCount = parseInt(inside.split('/')[0], 10);
    const crpsFinite = Number.isFinite(parseFloat(crps));
    const naiveVal = parseFloat(naive);
    console.log(`inside band=${inside} CRPS=${crps} vs weekly naive=${naive}`);
    if (!Number.isFinite(insideCount) || insideCount < 0 || insideCount > horizon) {
      throw new Error(`inside-band count out of [0, ${horizon}]: ${inside}`);
    }
    if (!crpsFinite) {
      throw new Error(`CRPS not finite: ${crps}`);
    }
    if (!Number.isFinite(naiveVal) || naiveVal <= 0) {
      throw new Error(`vs-weekly-naive ratio not finite/positive: ${naive}`);
    }
    report.metrics = { inside, crps, naive };

    // --- no range/native slider inputs anywhere on the page ---
    const inputCounts = await page.evaluate(() => ({
      total: document.querySelectorAll('input').length,
      range: document.querySelectorAll('input[type="range"]').length,
      color: document.querySelectorAll('input[type="color"]').length,
    }));
    console.log(`input elements: ${JSON.stringify(inputCounts)}`);
    if (inputCounts.total !== 0) {
      throw new Error(`page has ${inputCounts.total} <input> element(s), expected 0 (no native origin control)`);
    }

    // --- drag on the INPUT chart -> exactly one forecast, data unchanged, origin clamped ---
    const checksumBefore = await page.evaluate(() => window.__app.seriesChecksum());
    const before = await page.evaluate(() => window.__app.forecastCount);
    await page.locator('#inputChart').scrollIntoViewIfNeeded();
    const box = await page.locator('#inputChart').boundingBox();
    const y = box.y + box.height / 2;
    // drag well past the left edge -- must clamp, not error or run off-bounds
    await page.mouse.move(box.x + box.width * 0.9, y);
    await page.mouse.down();
    for (const frac of [0.6, 0.3, 0.05, 0.0]) {
      await page.mouse.move(box.x + box.width * frac, y);
      await page.waitForTimeout(20);
    }
    await page.mouse.up();
    await page.waitForFunction((n) => window.__app.forecastCount > n, before, { timeout: 10000 });
    await page.waitForTimeout(300); // settle in case of a stray second trigger
    const after = await page.evaluate(() => window.__app.forecastCount);
    const dragDelta = after - before;
    const originAfter = await page.evaluate(() => window.__app.origin());
    const bounds = await page.evaluate(() => window.__app.originBounds());
    const checksumAfter = await page.evaluate(() => window.__app.seriesChecksum());
    console.log(`drag: forecastCount ${before} -> ${after} (delta ${dragDelta}), origin -> ${originAfter}, bounds ${JSON.stringify(bounds)}`);
    console.log(`series checksum: ${checksumBefore} -> ${checksumAfter}`);
    if (dragDelta !== 1) {
      throw new Error(`drag produced ${dragDelta} forecasts, expected exactly 1`);
    }
    if (checksumAfter !== checksumBefore) {
      throw new Error(`drag changed the drawn series data (checksum ${checksumBefore} -> ${checksumAfter})`);
    }
    if (originAfter < bounds.lo || originAfter > bounds.hi) {
      throw new Error(`origin ${originAfter} not clamped within [${bounds.lo}, ${bounds.hi}]`);
    }

    report.pass = true;
    console.log(`\nPASS -- forecast ${ms.toFixed(1)} ms, inside band ${inside}, CRPS ${crps}, vs weekly naive ${naive} (backend: ${backend})`);
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
