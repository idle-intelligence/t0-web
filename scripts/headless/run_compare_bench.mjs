// Headless runner for bench/compare/index.html: drives the single-page
// head-to-head (onnxruntime-web vs t0-fast) through window.__app.run(),
// same Chromium build as the other scripts/headless runners, never the
// owner's browser. Prints the JSON result, screenshots the rendered page,
// and checks the acceptance gate: all latency numbers finite, agreement
// max-abs < 1% of range.
//
// Usage: node scripts/headless/run_compare_bench.mjs [--url http://127.0.0.1:8046/bench/compare/] [--context 512] [--horizon 32] [--quant q8_0] [--screenshot /path/to.png]
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
const url = args.url || 'http://127.0.0.1:8046/bench/compare/';
const context = args.context ? parseInt(args.context, 10) : 512;
const horizon = args.horizon ? parseInt(args.horizon, 10) : 32;
const quant = args.quant || 'q8_0';
const screenshot = args.screenshot || null;
const EXECUTABLE_PATH =
  '~/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
const page = await browser.newPage();
page.on('console', (msg) => console.log('[page]', msg.text()));
page.on('pageerror', (err) => console.error('[pageerror]', err.message));

console.log('Loading', url, { context, horizon, quant });
await page.goto(url, { waitUntil: 'load' });
console.log('running compare protocol (loads two models, may take a while)...');
const result = await page.evaluate(
  ([context, horizon, quant]) => window.__app.run({ context, horizon, quant }),
  [context, horizon, quant],
);
console.log(JSON.stringify(result, null, 2));

if (screenshot) {
  await page.screenshot({ path: screenshot, fullPage: true });
  console.log('screenshot saved to', screenshot);
}

// acceptance gate
const problems = [];
for (const row of result.latencyRows) {
  for (const k of ['downloadMB', 'cold', 'warmMedian', 'warmP90']) {
    if (!Number.isFinite(row[k])) problems.push(`${row.engine}.${k} not finite: ${row[k]}`);
  }
}
for (const row of result.batchRows) {
  if (row.error) problems.push(`batch ${row.engine}: ${row.error}`);
  else if (!Number.isFinite(row.msTotal) || !Number.isFinite(row.msPerSignal)) problems.push(`batch ${row.engine} not finite`);
}
if (!Number.isFinite(result.agreement.overall.maxAbsPct)) problems.push('agreement.overall.maxAbsPct not finite');
else if (result.agreement.overall.maxAbsPct >= 1.0) problems.push(`agreement max-abs ${result.agreement.overall.maxAbsPct.toFixed(3)}% of range >= 1%`);

if (problems.length) {
  console.error('GATE FAILED:\n' + problems.map((p) => '- ' + p).join('\n'));
  await browser.close();
  process.exit(1);
}
console.log('GATE PASSED: all rows finite, agreement max-abs', result.agreement.overall.maxAbsPct.toFixed(3) + '% of range < 1%');

await browser.close();
