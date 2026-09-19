// Headless smoke test for web/index.html: loads the page in Playwright's
// bundled Chromium (never the maintainer's real browser -- CLAUDE.md), waits
// for window.__app.ready, moves the forecast-origin slider through three
// origins via window.__app.setOrigin, and asserts each forecast is finite,
// monotone across quantile index (q0 <= q1 <= ... <= q_{n-1} at every
// timestep -- t0-alpha's trained quantile levels are sorted, so this must
// hold if the model/decoder are wired correctly), and horizon long.
//
// WASM CPU (burn-ndarray) only -- WebGPU is not wired up yet (see
// crates/t0-wasm/README.md), so there is nothing to select here.
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
const ORIGINS = (args.origins ?? '200,400,560').split(',').map((s) => parseInt(s.trim(), 10));
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

  const report = { url: URL_, origins: [], pass: false };

  try {
    console.log(`Loading ${URL_}`);
    const t0 = Date.now();
    await page.goto(URL_, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__app && window.__app.ready === true, undefined, { timeout: TIMEOUT_LOAD });
    const loadMs = Date.now() - t0;
    console.log(`Ready in ${loadMs} ms`);
    report.loadMs = loadMs;

    for (const origin of ORIGINS) {
      const result = await page.evaluate(async (o) => {
        window.__app.setOrigin(o);
        // Wait for this origin's forecast to land (setOrigin posts to the
        // worker; the result arrives async via the worker's onmessage).
        const start = performance.now();
        while (performance.now() - start < 30000) {
          const f = window.__app.lastForecast;
          if (f && f.origin === o) return f;
          await new Promise((r) => setTimeout(r, 50));
        }
        return null;
      }, origin);

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

      const originReport = { origin, ms, lengthOk, allFinite, monotoneOk, horizon, nQuantiles };
      report.origins.push(originReport);
      console.log(
        `origin=${origin}: ${ms.toFixed(1)} ms, length_ok=${lengthOk}, finite=${allFinite}, monotone=${monotoneOk}`
      );
      if (!lengthOk || !allFinite || !monotoneOk) {
        throw new Error(`origin ${origin} failed a gate: ${JSON.stringify(originReport)}`);
      }
    }

    report.pass = true;
    const avgMs = report.origins.reduce((s, o) => s + o.ms, 0) / report.origins.length;
    console.log(`\nPASS -- ${report.origins.length} origins, avg ${avgMs.toFixed(1)} ms/forecast (WASM CPU/ndarray; WebGPU pending)`);
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
