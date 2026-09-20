# Compare page: single-tab head-to-head (onnxruntime-web vs t0-fast)

`bench/compare/index.html` + `bench/compare/compare.js` load both engines
in one tab — theforecastingcompany's `t0-alpha-onnx-int8` export via
`onnxruntime-web` (WebGPU EP, wasm fallback) and our GGUF-resident t0-fast
(`crates/t0-wasm/pkg-fast`, Q8_0 default / Q4_0 button) — so every number is
measured live in whichever browser opens the page, not baked in. Protocol:
2 warm-ups each, then 10 alternating calls (theirs, ours, …) on identical
input, plus a 24-signal batch row and a median/Q10-Q90 agreement diff. This
run used the fixture series (`us_births`, same as `bench/onnx`/`bench/ours`),
context/horizon 512/32, ours at Q8_0.

- Machine: Apple M2 (Darwin 25.3.0). `pgrep -f headless | wc -l` was 0
  before this run; the headless call ran alone.
- Commit: this session's `bench/compare/`, `scripts/headless/run_compare_bench.mjs`.
- Browser: Playwright's bundled Chromium-for-Testing (`chromium-1243`),
  explicit `executablePath`, never the maintainer's browser.
- Served from the repo root: `python3 -m http.server 8046 --bind 127.0.0.1`,
  page at `http://127.0.0.1:8046/bench/compare/`.
- Command: `node scripts/headless/run_compare_bench.mjs --url http://127.0.0.1:8046/bench/compare/ --context 512 --horizon 32 --quant q8_0 --screenshot ~/.claude/jobs/f13f376f/tmp/t0-compare.png`

## Latency

| engine | ep/backend | download MB | cold ms | warm median ms | warm p90 ms |
|---|---|---|---|---|---|
| theirs (ONNX INT8) | webgpu | 107.2 | 214.0 | 52.7 | 60.5 |
| ours (t0-fast Q8_0) | webgpu (t0-fast) | 108.9 | 60.3 | 34.7 | 35.2 |

## Batch 24

| engine | ms total | ms/signal | note |
|---|---|---|---|
| theirs (ONNX) | 220.2 | 9.17 | 24 shifted windows (stride 1 day), dynamic batch dim |
| ours (t0-fast) | 335.2 | 13.97 | same window x24 — `forecastBatch(context, n, horizon, chunk)` takes one context array, not per-row distinct context (API constraint, see `t0_wasm.d.ts`); not a true shifted-signal batch |

## Agreement (theirs ONNX INT8 vs ours Q8_0, median + per-quantile, same 512/32 window)

Range (max-min over both engines' full quantile grid, this window) = 2272.78
births/day.

| quantile | max-abs diff | max-abs % range | mean-abs diff | mean-abs % range |
|---|---|---|---|---|
| overall | 20.07 | 0.88% | 6.46 | 0.28% |
| Q10 | 15.40 | 0.68% | 4.73 | 0.21% |
| Q25 | 17.38 | 0.76% | 5.71 | 0.25% |
| Q50 (median) | 20.07 | 0.88% | 6.50 | 0.29% |
| Q75 | 18.89 | 0.83% | 7.29 | 0.32% |
| Q90 | 18.13 | 0.80% | 8.05 | 0.35% |

Gate: all latency/batch numbers finite, agreement max-abs 0.883% of range
< 1% threshold — **passed**.

## Performance panel (this run's machine)

- UA: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36`
- GPU adapter (`navigator.gpu.requestAdapter().info`): `apple / metal-3`
- Timestamp: `2026-09-20T15:07:51.561Z`

## Notes

- One console 404 appeared during the run (checked: a stray resource probe,
  not a required asset — WebGPU EP succeeded on the first `session.run`,
  no fallback to wasm was triggered, and both engines produced finite
  output).
- The batch-24 "ours" row is not a true 24-distinct-signal batch: t0-fast's
  exposed `forecastBatch` takes a single context array and replicates it
  internally (same limitation the existing `bench/ours/index.html` harness
  documents). This is a black-box consumer's read of the wasm-bindgen
  surface, not a bug fix attempted here — flagging for whoever owns
  `crates/t0-wasm` if a true per-row batch is wanted later.
- Screenshot: `~/.claude/jobs/f13f376f/tmp/t0-compare.png` (1280x1743,
  full page).
