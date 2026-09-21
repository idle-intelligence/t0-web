// Headless runner for bench/compare/index.html: drives the single-page
// head-to-head (onnxruntime-web vs t0-fast) through window.__app.run(),
// same Chromium build as the other scripts/headless runners, never the
// owner's browser. Prints the JSON result, screenshots the rendered page,
// and checks the acceptance gate: all latency numbers finite, agreement
// max-abs < 1% of range.
//
// Usage: node scripts/headless/run_compare_bench.mjs [--url http://127.0.0.1:8046/bench/compare/] [--context 512] [--horizon 32] [--quant q8_0] [--screenshot /path/to.png]
import { chromium } from 'playwright';
import os from 'node:os';
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
const url = args.url || 'http://127.0.0.1:8046/bench/compare/';
const context = args.context ? parseInt(args.context, 10) : 512;
const horizon = args.horizon ? parseInt(args.horizon, 10) : 32;
const quant = args.quant || 'q8_0';
const useF32 = !!args.f32;
const screenshot = args.screenshot || null;
const EXECUTABLE_PATH =
  process.env.CHROMIUM_PATH ||
  path.join(os.homedir(), 'Library/Caches/ms-playwright', 'chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing');

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
const page = await browser.newPage();
page.on('console', (msg) => console.log('[page]', msg.text()));
page.on('pageerror', (err) => console.error('[pageerror]', err.message));

console.log('Loading', url, { context, horizon, quant, useF32 });
await page.goto(url, { waitUntil: 'load' });
console.log('running compare protocol (loads two models, may take a while)...');
const result = await page.evaluate(
  ([context, horizon, quant, useF32]) => window.__app.run({ context, horizon, quant, useF32 }),
  [context, horizon, quant, useF32],
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
const agreeKeys = useF32 ? ['oursVsF32', 'theirsVsF32', 'theirsVsOurs'] : ['theirsVsOurs'];
for (const key of agreeKeys) {
  const pct = result.agreement[key].overall.maxAbsPct;
  if (!Number.isFinite(pct)) problems.push(`agreement.${key}.overall.maxAbsPct not finite`);
}
const theirsVsOursPct = result.agreement.theirsVsOurs.overall.maxAbsPct;
if (Number.isFinite(theirsVsOursPct) && theirsVsOursPct >= 1.0) problems.push(`agreement theirsVsOurs max-abs ${theirsVsOursPct.toFixed(3)}% of range >= 1%`);
// Relative gate, not absolute: max-abs / (max-min of the compared
// quantile-grid values) <= 1e-5. An absolute 1e-4 gate was calibrated
// against a small-magnitude synthetic fixture and falsely failed on
// real-world thousands-magnitude data (0.001953125 absolute there is
// ~1e-6 relative) -- see docs/runs/2026-09-20-compare-page.md.
if (result.rowParityMaxAbs == null || !Number.isFinite(result.rowParityMaxAbs)) problems.push(`forecastBatchRows parity max-abs not finite: ${result.rowParityMaxAbs}`);
else if (result.rowParityRelative == null || !Number.isFinite(result.rowParityRelative)) problems.push(`forecastBatchRows parity relative not finite: ${result.rowParityRelative}`);
else if (result.rowParityRelative > 1e-5)
  problems.push(`forecastBatchRows parity max-abs ${result.rowParityMaxAbs} / range ${result.rowParityRange} = ${result.rowParityRelative} > 1e-5`);

if (problems.length) {
  console.error('GATE FAILED:\n' + problems.map((p) => '- ' + p).join('\n'));
  await browser.close();
  process.exit(1);
}
const extra = useF32
  ? [', ours-vs-F32 max-abs', result.agreement.oursVsF32.overall.maxAbs.toExponential(3)]
  : [', F32 reference opt-in was off'];
console.log(
  'GATE PASSED: all rows finite, agreement theirsVsOurs max-abs',
  theirsVsOursPct.toFixed(3) + '% of range < 1%',
  ...extra,
  ', forecastBatchRows parity max-abs',
  result.rowParityMaxAbs.toExponential(3),
  '/ range',
  result.rowParityRange,
  '=',
  result.rowParityRelative.toExponential(3),
  '<= 1e-5',
);

await browser.close();
