// Captures a Chrome DevTools performance trace of our t0-wasm WebGPU bench
// (bench/ours/index.html), one for a single warm forecast, one for a
// 24-signal batch. Uses CDP Tracing directly (page.context().newCDPSession)
// with categories disabled-by-default-devtools.timeline, gpu,
// blink.user_timing -- Playwright's page.tracing wraps a narrower default
// category set, so CDP is used directly to get GPU dispatch events.
//
// Usage: node scripts/headless/trace_ours.mjs --url http://127.0.0.1:8033/ --out-dir /path
import { chromium } from 'playwright';
import os from 'node:os';
import path from 'node:path';
import fs from 'fs';

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
const outDir = args['out-dir'] || '/tmp';
const pkg = args.pkg || './pkg-wgpu';
const gguf = args.gguf || './t0-alpha-q8_0.gguf';
const EXECUTABLE_PATH =
  process.env.CHROMIUM_PATH ||
  path.join(os.homedir(), 'Library/Caches/ms-playwright', 'chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing');

const CATEGORIES = [
  'disabled-by-default-devtools.timeline',
  'gpu',
  'blink.user_timing',
  'disabled-by-default-devtools.timeline.frame',
];

async function traceOne(page, client, label, fn, outFile) {
  await client.send('Tracing.start', {
    categories: CATEGORIES.join(','),
    transferMode: 'ReturnAsStream',
  });
  await fn();
  const events = [];
  const done = new Promise((resolve) => {
    client.on('Tracing.tracingComplete', async (params) => {
      resolve(params);
    });
  });
  await client.send('Tracing.end');
  const params = await done;
  if (params.stream) {
    let data = '';
    while (true) {
      const chunk = await client.send('IO.read', { handle: params.stream });
      data += chunk.data;
      if (chunk.eof) break;
    }
    fs.writeFileSync(outFile, data);
  }
  console.log(`[trace] ${label} written to ${outFile}`);
}

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH });
const page = await browser.newPage();
page.on('console', (msg) => console.log('[page]', msg.text()));
page.on('pageerror', (err) => console.error('[pageerror]', err.message));

console.log('Loading', url);
await page.goto(url, { waitUntil: 'load' });

// warm the model up (load + a few forecasts) BEFORE tracing, so the trace
// only captures steady-state dispatch behavior, not shader compile/autotune.
await page.evaluate(([pkg, gguf]) => window.__bench.warmup(pkg, gguf), [pkg, gguf]);

const client = await page.context().newCDPSession(page);

await traceOne(page, client, 'single warm forecast', async () => {
  await page.evaluate(() => window.__bench.oneForecast());
}, `${outDir}/t0-trace-single.json`);

await traceOne(page, client, 'batch-24', async () => {
  await page.evaluate(() => window.__bench.oneBatch());
}, `${outDir}/t0-trace-batch.json`);

await browser.close();
