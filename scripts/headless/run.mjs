// Headless smoke test for web/index.html: loads the page in Playwright's
// bundled Chromium (never the maintainer's real browser -- CLAUDE.md).
//
// Covers:
//   1. three origins via window.__app.setOrigin -- forecast finite, monotone
//      across quantile index, horizon-long.
//   2. a real mouse drag on the #chart canvas produces exactly one forecast
//      (debounced, commit-on-release -- not one per pointermove).
//   3. MAE/sMAPE (computed against the real continuation) are finite.
//   4. moving the origin slider without releasing it ('input' events only)
//      does not grow the STATUS log line count (repeated forecast/download
//      lines update in place; only genuine state changes append).
//   5. the STATUS block (two fixed lines, collapsed log) has constant
//      height across 20 forecasts at different origins.
//
// Backend is auto-selected by web/worker.js (WebGPU if navigator.gpu
// exists, else WASM CPU/burn-ndarray) -- this bundled Chromium-for-Testing
// exposes navigator.gpu with a real hardware adapter on http origins with
// no extra launch flags needed (confirmed this session on this Mac; see
// docs/runs/2026-09-19-web-smoke.md), so this script always exercises
// whichever backend that Chromium picks and logs it.
//
// Usage: node scripts/headless/run.mjs [--url http://127.0.0.1:PORT/] [--origins 200,400,560]
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
const ORIGINS = (args.origins ?? '500,3600,7000').split(',').map((s) => parseInt(s.trim(), 10));
const TIMEOUT_LOAD = parseInt(args['timeout-load'] ?? String(5 * 60 * 1000), 10);

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const { chromium } = await import('playwright');

// Same cached Chromium-for-Testing binary this session verified works
// standalone (`node -e "require('playwright').chromium.launch(...)"`) --
// passed explicitly so this doesn't depend on playwright's own browser
// revision bookkeeping matching whatever happens to be cached.
const EXECUTABLE_PATH =
  '~/Library/Caches/ms-playwright/chromium-1229/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

async function waitForForecast(page, origin, timeoutMs = 30000) {
  return page.evaluate(async ({ o, timeoutMs }) => {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
      const f = window.__app.lastForecast;
      if (f && f.origin === o) return f;
      await new Promise((r) => setTimeout(r, 50));
    }
    return null;
  }, { o: origin, timeoutMs });
}

async function main() {
  const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
  const page = await browser.newPage();

  const consoleLines = [];
  page.on('console', (msg) => consoleLines.push(msg.text()));
  page.on('pageerror', (err) => consoleLines.push(`[pageerror] ${err.message}`));

  const report = { url: URL_, origins: [], pass: false };

  try {
    console.log(`Loading ${URL_}`);
    const t0 = Date.now();
    await page.goto(URL_, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__app && window.__app.ready === true, undefined, { timeout: TIMEOUT_LOAD });
    const loadMs = Date.now() - t0;
    console.log(`Ready in ${loadMs} ms`);
    report.loadMs = loadMs;

    // Wait for the initial (page-load-default-origin) forecast to settle
    // before starting the timed checks below.
    await page.waitForFunction(() => window.__app.lastForecast !== null, undefined, { timeout: 30000 });
    const backend = await page.evaluate(() => document.getElementById('metricBackend').textContent);
    console.log(`Backend: ${backend}`);
    report.backend = backend;

    for (const origin of ORIGINS) {
      await page.evaluate((o) => window.__app.setOrigin(o), origin);
      const result = await waitForForecast(page, origin);
      if (!result) {
        throw new Error(`origin ${origin}: no forecast arrived within 30s`);
      }

      const { quantiles, nQuantiles, horizon, ms } = result;
      const lengthOk = quantiles.length === horizon * nQuantiles;
      const allFinite = quantiles.every((v) => Number.isFinite(v));
      let monotoneOk = true;
      for (let t = 0; t < horizon && monotoneOk; t++) {
        for (let q = 1; q < nQuantiles; q++) {
          if (quantiles[t * nQuantiles + q] < quantiles[t * nQuantiles + q - 1] - 1e-4) {
            monotoneOk = false;
            break;
          }
        }
      }

      const { mae, smape } = await page.evaluate(() => ({
        mae: document.getElementById('metricMae').textContent,
        smape: document.getElementById('metricSmape').textContent,
      }));
      const metricsFinite = Number.isFinite(parseFloat(mae)) && Number.isFinite(parseFloat(smape));

      const originReport = { origin, ms, lengthOk, allFinite, monotoneOk, metricsFinite, mae, smape, horizon, nQuantiles };
      report.origins.push(originReport);
      console.log(
        `origin=${origin}: ${ms.toFixed(1)} ms, length_ok=${lengthOk}, finite=${allFinite}, monotone=${monotoneOk}, MAE=${mae}, sMAPE=${smape}%`
      );
      if (!lengthOk || !allFinite || !monotoneOk || !metricsFinite) {
        throw new Error(`origin ${origin} failed a gate: ${JSON.stringify(originReport)}`);
      }
    }

    // --- drag -> exactly one forecast ---
    const before = await page.evaluate(() => window.__app.forecastCount);
    await page.locator('#chart').scrollIntoViewIfNeeded();
    const box = await page.locator('#chart').boundingBox();
    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + box.width * 0.3, y);
    await page.mouse.down();
    for (const frac of [0.35, 0.4, 0.45, 0.5, 0.55]) {
      await page.mouse.move(box.x + box.width * frac, y);
      await page.waitForTimeout(20);
    }
    await page.mouse.up();
    // debounce is 150ms; give it margin, then let the forecast land.
    await page.waitForFunction(
      (n) => window.__app.forecastCount > n,
      before,
      { timeout: 10000 }
    );
    await page.waitForTimeout(300); // settle window in case of a second stray trigger
    const after = await page.evaluate(() => window.__app.forecastCount);
    const dragDelta = after - before;
    console.log(`drag: forecastCount ${before} -> ${after} (delta ${dragDelta})`);
    if (dragDelta !== 1) {
      throw new Error(`drag produced ${dragDelta} forecasts, expected exactly 1`);
    }

    // --- slider 'input' (no release) must not grow the log ---
    const logBefore = await page.evaluate(() => window.__app.logLineCount());
    await page.evaluate(() => {
      const slider = document.getElementById('originSlider');
      const base = parseInt(slider.value, 10);
      for (const delta of [10, 20, 30, 20, 10]) {
        slider.value = base + delta;
        slider.dispatchEvent(new Event('input'));
      }
    });
    await page.waitForTimeout(200);
    const logAfter = await page.evaluate(() => window.__app.logLineCount());
    console.log(`slider drag (input only): log lines ${logBefore} -> ${logAfter}`);
    if (logAfter !== logBefore) {
      throw new Error(`slider 'input' events grew the log (${logBefore} -> ${logAfter}), expected no change`);
    }

    // --- STATUS block height must stay constant across 20 forecasts ---
    const heightBefore = await page.evaluate(() => window.__app.statusBlockHeight());
    const lo = parseInt(await page.evaluate(() => document.getElementById('originSlider').min), 10);
    const hi = parseInt(await page.evaluate(() => document.getElementById('originSlider').max), 10);
    const heights = [heightBefore];
    for (let i = 0; i < 20; i++) {
      const o = lo + Math.floor(((hi - lo) * i) / 19);
      await page.evaluate((oo) => window.__app.setOrigin(oo), o);
      await waitForForecast(page, o);
      heights.push(await page.evaluate(() => window.__app.statusBlockHeight()));
    }
    const minH = Math.min(...heights), maxH = Math.max(...heights);
    console.log(`status block height over 20 forecasts: ${minH.toFixed(1)}-${maxH.toFixed(1)}px`);
    if (maxH - minH > 0.5) {
      throw new Error(`STATUS block height varied (${minH.toFixed(1)}-${maxH.toFixed(1)}px) across 20 forecasts`);
    }

    report.pass = true;
    const avgMs = report.origins.reduce((s, o) => s + o.ms, 0) / report.origins.length;
    console.log(`\nPASS -- ${report.origins.length} origins, avg ${avgMs.toFixed(1)} ms/forecast (backend: ${report.backend})`);
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
