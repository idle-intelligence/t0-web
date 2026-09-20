# Head-to-head: official t0-alpha ONNX INT8 export vs ours (same browser, same machine)

- Machine: Apple M2 (Darwin 25.3.0). GPU rule checked before every run:
  `pgrep -f 'cargo (build|test)|wasm-pack|rustc'` — one unrelated `llm-life`
  native release build was compiling in a sibling repo during the wasm-pack
  builds below (compile-only, no GPU device touched); no other process held
  the GPU device during any headless run (webgpu bench calls run
  sequentially, one at a time, never concurrently).
- Commit: `09a7ed9` + this run's uncommitted changes (`t0-cli forecast-raw`,
  `t0_core::forecast_batch_chunked_async`, `T0Wasm::forecastBatch`,
  `bench/onnx/`, `bench/ours/`).
- Browser: Playwright's bundled Chromium-for-Testing, `chromium-1243`
  (`Google Chrome for Testing 153.0.8010.12`), launched with an explicit
  `executablePath`, never the maintainer's browser.
- `onnxruntime-web`: `1.29.0` (pinned to match the artifact's own validation
  card, `manifest.json`'s `runtime.onnxruntime_web`), installed via
  `npm install --save-dev onnxruntime-web@1.29.0` — a local copy is served
  from `bench/onnx/` (`ort.webgpu.min.js` + the `ort-wasm-simd-threaded.*`
  wasm/mjs files copied from `node_modules/onnxruntime-web/dist/`), no CDN
  at test time.
- wgpu backend: `burn-wgpu` / `burn` `0.20` (`Cargo.toml`), `cubecl-wgpu`
  transitively (pinned pre-1.0 versions per this repo's `CLAUDE.md`).
- Artifact under test (theirs): `theforecastingcompany/t0-alpha-onnx-int8`,
  `t0-alpha-grouped-int8.onnx`, 107,151,882 bytes, already present at
  `~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha-onnx-int8/`
  (not re-downloaded). Single file, no external-data shard.
- Artifacts under test (ours): `web/models/t0-alpha-q8_0.gguf`
  (108,936,047 bytes) and `/tmp/t0-alpha-q4_0.gguf` (58,604,399 bytes),
  both this session's `t0-cli export-gguf` output from
  `theforecastingcompany/t0-alpha`'s `model.safetensors` (406.6 MB F32),
  same checkpoint as the ONNX export (`manifest.json`'s
  `source.model: theforecastingcompany/t0-alpha`).
- Fixture: `web/data/series.f32` (`us_births`, daily, 7305 points), same
  series `web/worker.js` uses. Origin index 3600 (1978-11-14), context 512
  (`origin - 512 .. origin`), horizon 32 — identical window to
  `docs/BENCHMARKS.md`'s single-signal latency tables and
  `docs/runs/2026-09-19-web-smoke.md`'s `origin=3600` row.

## What changed in the repo to run this

- `crates/cli/src/main.rs`: new `forecast-raw` subcommand (`--series-file`,
  `--origin`, reuses `--context`/`--horizon`/`--weights`/`--config`) — the
  existing `forecast`/`gifteval` subcommands only take fixture-manifest or
  GIFT-Eval-manifest context files, not an arbitrary raw series slice, so
  there was no existing way to get an F32 reference forecast for this
  exact window. Writes the raw `horizon * n_quantiles` f32 output to a
  file, same layout as `gifteval`'s per-window output.
- `crates/t0-core/src/model.rs`: `forecast_batch_chunked_async`, an async
  (`into_data_async().await`) twin of the existing `forecast_batch_chunked`
  — needed because `t0-wasm`'s `wgpu` feature can never call the sync
  `forecast_batch_chunked` (deadlocks in a real browser, same reasoning as
  the existing `forecast_async`/`forecast`).
- `crates/t0-wasm/src/lib.rs`: `T0Wasm::forecastBatch(context, n_signals,
  horizon, chunk_size)` — exposes the above so the "24-signal batch" number
  could be measured for our side too (the wasm-bindgen surface previously
  only exposed single-signal `forecast`).
- `cargo clippy -p t0-core -p t0-wasm --release -- -D warnings`: clean.
  Both `wasm-pack build crates/t0-wasm --target web --release` and
  `... --out-dir pkg-wgpu --release -- --no-default-features --features
  wgpu` built clean.
- `bench/onnx/index.html` + `scripts/headless/run_onnx_bench.mjs`: loads
  the official ONNX export through `onnxruntime-web`, WebGPU EP first
  (with the artifact's own `webgpu-options.js` `forceCpuNodeNames` list —
  copied inline into the harness, not re-fetched), falls back to `wasm` on
  failure. Reports cold (first `session.run`, includes graph
  init/compile), warm (median of 10), and a 24-row batch (all rows given
  the same context, matching `t0.run`'s observed default desktop batch
  group size of 24 from `docs/reports/t0-published-numbers.md`).
- `bench/ours/index.html` + `scripts/headless/run_ours_bench.mjs`: same
  protocol against `t0-wasm`'s `pkg`/`pkg-wgpu` builds, one page load per
  (backend, quant) combination so nothing is shared/warmed across rows.

## Commands (exact)

```
# F32 reference (native, ndarray)
cargo build -p t0-cli --release
./target/release/t0-cli forecast-raw \
  --weights ../models/hf/theforecastingcompany/t0-alpha/model.safetensors \
  --config ../models/hf/theforecastingcompany/t0-alpha/config.json \
  --series-file web/data/series.f32 --origin 3600 --context 512 --horizon 32 \
  --out /tmp/t0_f32_ref_origin3600.f32

# their side
npm install --save-dev onnxruntime-web@1.29.0
cd bench/onnx && python3 -m http.server 8032 &
node scripts/headless/run_onnx_bench.mjs --url http://127.0.0.1:8032/

# our side
wasm-pack build crates/t0-wasm --target web --release
wasm-pack build crates/t0-wasm --target web --out-dir pkg-wgpu --release -- --no-default-features --features wgpu
cd bench/ours && python3 -m http.server 8033 &
node scripts/headless/run_ours_bench.mjs --url http://127.0.0.1:8033/ --pkg ./pkg-wgpu --gguf ./t0-alpha-q8_0.gguf --label "Q8_0 WebGPU"
node scripts/headless/run_ours_bench.mjs --url http://127.0.0.1:8033/ --pkg ./pkg-wgpu --gguf ./t0-alpha-q4_0.gguf --label "Q4_0 WebGPU"
node scripts/headless/run_ours_bench.mjs --url http://127.0.0.1:8033/ --pkg ./pkg     --gguf ./t0-alpha-q8_0.gguf --label "Q8_0 CPU"
```

## Which execution provider ran, theirs

**WebGPU.** `bench/onnx/index.html` requests the `webgpu` EP (with the
artifact's own `forceCpuNodeNames` override list) first and only falls
back to `wasm` if session creation or the first `run()` throws. Neither
happened — `results.ep === 'webgpu'` on every run, no fallback triggered,
no console errors. This Chromium (`chromium-1243`) exposes a real hardware
Metal adapter over `http://` by default, same finding as
`docs/runs/2026-09-19-web-smoke.md` for our own WebGPU path.

## Raw numbers

| metric | their ONNX INT8 (webgpu EP) | our Q8_0 WebGPU | our Q4_0 WebGPU | our Q8_0 CPU (ndarray/wasm) |
|---|---|---|---|---|
| download bytes | 107,151,882 | 108,936,047 | 58,604,399 | 108,936,047 |
| download MB | 107.2 | 108.9 | 58.6 | 108.9 |
| cold (ms, first call) | 167.5 | 707.1 | 276.6 | 604.9 |
| warm median of 10 (ms) | 64.0 | 162.5 | 162.7 | 576.4 |
| 24-row/signal batch (ms total) | 212.9 | 1105.6 | 803.8 | 7693.7 |
| 24-row/signal batch (ms/signal) | 8.87 | 46.07 | 33.49 | 320.57 |

Warm raw reps (ms), all 10 per row:

- theirs (webgpu): `[81.2, 79.7, 79.8, 63.0, 61.5, 62.7, 60.4, 65.0, 65.5, 60.5]`
- ours Q8_0 webgpu: see `/tmp/ours_q8_webgpu.json` (this session, not committed — file MB/ms summarized above)
- ours Q4_0 webgpu: see `/tmp/ours_q4_webgpu.json`
- ours Q8_0 cpu: see `/tmp/ours_q8_cpu.json`

## Verification: ONNX INT8 output vs our F32 reference, same window

Reference: `/tmp/t0_f32_ref_origin3600.f32` (`t0-cli forecast-raw`, F32
weights, `ndarray` backend), `us_births`, origin 3600, context 512, horizon
32 — 160 values (32 timesteps x 5 trained quantile levels
`[0.1, 0.25, 0.5, 0.75, 0.9]`), forecast range (max-min over these 160
values) = 2266.84 (births/day units).

| output | max-abs diff | max-abs as % of range | mean-abs diff | mean-abs as % of range |
|---|---|---|---|---|
| their ONNX INT8 (webgpu) | 17.99 | 0.79% | 5.69 | 0.25% |
| our Q8_0 (webgpu, same as cpu — compute is F32 on both) | 4.47 | 0.20% | 1.73 | 0.08% |
| our Q4_0 (webgpu) | 86.94 | 3.84% | 21.03 | 0.93% |

On this single window: our Q8_0 is closer to the F32 reference than their
INT8 export (0.20% vs 0.79% max-abs-as-%-of-range) and smaller (108.9 MB
vs 107.2 MB is roughly a wash — 1.6% larger). Our Q4_0 is looser than
their INT8 (3.84% vs 0.79%) but half the size. This is one window, not a
statistically powered comparison — see the 54-case drift benchmark below
for the aggregate picture, and note this window's absolute drift numbers
differ from that benchmark's (different case, different denominator) by
construction, not contradiction.

**Caveat on the debugging path to this number**: the first diff attempt
gave a nonsense max-abs diff (~98% of range) because `Buffer.buffer` from
Node's `fs.readFileSync` on a small file is a view into Node's shared
buffer pool with a nonzero `byteOffset` — `new Float32Array(buf.buffer)`
without `buf.byteOffset` silently read from the wrong offset. Fixed with
`new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)`. Not a
bug in `t0-cli forecast-raw` or the ONNX harness; a bug in the one-off
Node comparison script, caught by the values-should-roughly-agree sanity
check before trusting the number.

## 54-case drift benchmark, for context (from `docs/BENCHMARKS.md`, already run this weekend)

This is our own case generator (their generator isn't published, see
`docs/reports/t0-published-numbers.md`), t0-alpha weights, drift formula
matching their card verbatim. `int8-theirs` is *our reimplementation* of
their per-channel INT8 recipe (96 matmuls INT8, 6 I/O projections FP32),
not the real `.onnx` artifact scored above — the real-artifact number is
the single-window verification table above, which is the more authoritative
one for "their shipped export," this 54-case table is the more
statistically powered one for "their recipe on our reimplementation":

| quant | mean drift worst % | mean drift mean % |
|---|---|---|
| our Q8_0 | 0.5299 | 0.1076 |
| our Q4_0 | 3.1084 | 1.3816 |
| int8-theirs (our reimplementation of their recipe) | 0.8494 | 0.1774 |

## GIFT-Eval subset CRPS/MASE

`docs/runs/2026-09-19-official-protocol-subset.md` (named in this task's
brief) **does not exist** — the actual subset run and its numbers live in
`docs/BENCHMARKS.md`'s "GIFT-Eval subset" section
(`docs/runs/2026-09-19-gifteval-subset.md` is the real analysis doc for
it). Flagging the stale filename rather than silently substituting.

That subset (8 configs / 382 windows, capped 512-context, our own port
only — reference PyTorch F32 vs our F32/Q8_0/Q4_0) has aggregate numbers:

| | reference F32 | our F32 | our Q8_0 | our Q4_0 |
|---|---|---|---|---|
| CRPS (aggregate) | 0.0897 | 0.0897 | 0.0897 | 0.0898 |
| MASE (aggregate) | 1.2139 | 1.2139 | 1.2139 | 1.2157 |

**This does not include the ONNX INT8 artifact** — running the official
export through the full `gluonts`-based GIFT-Eval harness (Python,
`onnxruntime` CPU EP, per `onnx_predictor.py` in the `fullsuite` worktree)
is a separate, longer task not attempted this session (that predictor
requires the `fullsuite` worktree's own venv/dependency stack, which this
task was scoped to leave untouched). The table in `docs/BENCHMARKS.md`
below marks this cell N/A rather than inferring it from the published
full-protocol number (different context length, different config set —
not comparable, see `docs/BENCHMARKS.md`'s existing caveat on this).

## What this doesn't cover

- WASM-only fallback timing for their export (WebGPU ran, so no comparison
  needed per this task's brief).
- A true multi-signal batched kernel comparison beyond n=24 — not
  requested this session.
- Phone/mobile timing (`t0.run`'s own batch-size halving is desktop vs
  mobile, not exercised here — this is all "desktop" Chromium-for-Testing
  on the M2).
