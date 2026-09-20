# Compare page: single-tab head-to-head (onnxruntime-web vs t0-fast)

`bench/compare/index.html` + `bench/compare/compare.js` load both engines
in one tab — theforecastingcompany's `t0-alpha-onnx-int8` export via
`onnxruntime-web` (WebGPU EP, wasm fallback) and our GGUF-resident t0-fast
(`crates/t0-wasm/pkg-fast`, Q8_0 default / Q4_0 button) — so every number is
measured live in whichever browser opens the page, not baked in. This run
also loads a third, F32-residency reference build (`T0Wasm.loadF32`
against the raw `model.safetensors`) to check the two quantized builds
against true ground truth, not just against each other. Protocol: 2
warm-ups each, then 10 alternating calls (theirs, ours, …) on identical
input, a 24-signal batch row (now 24 *distinct* shifted windows on both
engines, via `forecastBatchRows`), a `forecastBatchRows`-vs-`forecast()`
per-row parity check, and a three-way median/Q10-Q90 agreement diff
(theirs-vs-ours, ours-vs-F32, theirs-vs-F32). Fixture series (`us_births`,
same as `bench/onnx`/`bench/ours`), context/horizon 512/32, ours at Q8_0.

- Machine: Apple M2 (Darwin 25.3.0), hot/sluggish tonight — **no
  latency numbers from this run**; only the timing-independent
  correctness numbers below were captured. Latency/batch ms rerun is
  pending a cold machine.
- Commit: this session's `bench/compare/`, `crates/t0-fast`,
  `crates/t0-wasm`, `crates/t0-core`, `scripts/headless/run_compare_bench.mjs`
  (see git log around this doc's commit).
- Browser: Playwright's bundled Chromium-for-Testing (`chromium-1243`),
  explicit `executablePath`, never a personal browser.
- Served from the repo root: `python3 -m http.server 8046 --bind 127.0.0.1`,
  page at `http://127.0.0.1:8046/bench/compare/`.
- Command: `node scripts/headless/run_compare_bench.mjs --url http://127.0.0.1:8046/bench/compare/ --context 512 --horizon 32 --quant q8_0`

## Latency / Batch 24

**Pending, cold-machine run.** The laptop was hot and sluggish for this
session; ms numbers from tonight's run would not be representative and
are not recorded here. Re-run needed:

| engine | warm ms | batch-24 ms/signal |
|---|---|---|
| theirs (ONNX INT8) | pending, cold-machine run | pending, cold-machine run |
| ours (t0-fast Q8_0) | pending, cold-machine run | pending, cold-machine run |

## forecastBatchRows row parity (timing-independent)

Every one of the 24 distinct-window batch rows, diffed against that same
row's own single-signal `forecast()` call, same Q8_0 model:

**max-abs diff across all 24 rows: 0.001953125** (birth-count units; the
window's full quantile-grid range this run was ~2272.78, so this is
~0.086% of range). This is **above the 1e-4 absolute gate** the headless
script checks (`GATE FAILED`), reproduced identically (same exact
0.001953125 value) across two consecutive runs, so it is deterministic,
not sampling noise. Read as GPU batch-dispatch floating-point
non-associativity, not a logic bug: `forecast_batch_rows_chunked_async`'s
per-row output is structurally guaranteed to touch the same weights/data
in the same patch grouping as a standalone call (see its doc comment in
`crates/t0-fast/src/lib.rs`), but a `v=24` batched forward pass and a
`v=1` forward pass are different GEMM/reduction shapes on the GPU, so
exact bit-for-bit equality isn't guaranteed at this data's magnitude
(hundreds to thousands, so float32's ~7-digit precision alone yields an
absolute epsilon in the 1e-3 range) — this repo's existing native
`chunk_wgpu.rs` batch-vs-single test allows 1e-5 absolute at its
fixture's ~1-magnitude data, which scales to the same order as this
result once magnitude is accounted for. The 1e-4 absolute threshold this
session picked was calibrated against a small-magnitude synthetic fixture
(`crates/t0-fast/tests/batch_rows.rs`, values in `[-1, 1]`) and is not the
right absolute bound for real-world-magnitude data; a relative or
magnitude-scaled threshold is the correct follow-up, not a code fix to
`forecastBatchRows` itself.

## Agreement — theirs vs ours (valid)

Range (max-min over both engines' full quantile grid, this window) =
2272.78 births/day.

| quantile | max-abs diff | max-abs % range | mean-abs diff | mean-abs % range |
|---|---|---|---|---|
| overall | 20.07 | 0.88% | 6.46 | 0.28% |
| Q10 | 15.40 | 0.68% | 4.73 | 0.21% |
| Q25 | 17.38 | 0.76% | 5.71 | 0.25% |
| Q50 (median) | 20.07 | 0.88% | 6.50 | 0.29% |
| Q75 | 18.89 | 0.83% | 7.29 | 0.32% |
| Q90 | 18.13 | 0.80% | 8.05 | 0.35% |

Unchanged from the prior compare-page run (same order of magnitude, same
gate: theirs-vs-ours max-abs 0.883% of range < 1% — **passed**).

## Agreement — ours vs F32 reference, theirs vs F32 reference (INVALID this run — see below)

Raw numbers this run (range = 2687.54 births/day, a different window's
full-grid range than the theirs-vs-ours pair above because the F32
comparison spans a different value range):

| comparison | overall max-abs | overall max-abs % range |
|---|---|---|
| ours (Q8_0) vs F32 | 1943.36 | 72.31% |
| theirs (INT8) vs F32 | 1945.72 | 72.40% |

**These two rows are not trustworthy and are not the real F32 divergence.**
72% of range is not a plausible quantization error for either engine (the
theirs-vs-ours row above shows both quantized builds agree with each
other to under 1%, which would be impossible if either diverged this far
from a shared F32 ancestor). Root cause found while writing this doc, not
yet fixed: `bench/compare` now loads **two** `T0Wasm`/`GpuModel`
instances (the Q8_0 build and the new F32-reference build) that share one
process-wide `t0_fast::Engine`, and therefore one `Pool`
(`crates/t0-fast/src/pool.rs`). `Pool::bind_group`'s cache key
(`crates/t0-fast/src/model.rs`'s `linear()`, e.g. `"layer0.qkv.tiled"`) is
a bare per-call-site string with no per-model or per-weight-quant-type
component, and cache entries are invalidated only by a *global*
allocation-generation counter that the pool's own working buffers bump —
model *weight* buffers are allocated outside the pool and never bump it.
When the F32 model's first forward call reuses the exact same
`(v, p)` shape the Q8_0 model already ran at, every `linear()` call site
hits the cache with the *previous* (Q8_0-built) `wgpu::BindGroup` — which
is bound to Q8_0's actual weight buffers and Q8_0's compute pipeline
layout, not F32's — regardless of the fresh `entries` the F32 call
constructs. This predates this session's F32 work (the bug is in shared
`pool.rs`/`model.rs` code), but this session's `T0Wasm::loadF32` addition
is the first caller in this codebase's history to run two differently-
quantized `GpuModel`s against one shared `Engine`, so it's the first
thing to expose it. Fix is architectural (give each `GpuModel` its own
`Pool`, or fold weight-buffer/quant-type identity into the bind-group
cache key) and was not attempted in this session — flagged here rather
than shipped as a false "our engine differs from F32 by 72%" number.

Gate: theirs-vs-ours passed (0.883% < 1%); forecastBatchRows row parity
failed the session's 1e-4 absolute threshold (see above, read as a
threshold-calibration issue, not a functional bug); ours-vs-F32 and
theirs-vs-F32 are blocked on the Pool-sharing bug above and are not
gated this run.

## Fixed in 89080e8 / 453e1d1 / follow-up session

Both follow-ups from this doc's priority list are done: `89080e8` gives
every `GpuModel` its own `Pool` (Engine's device/queue/pipelines stay
shared, its `Pool` field is gone), and `453e1d1` makes the
`forecastBatchRows` parity gate relative (max-abs / range <= 1e-5)
instead of the miscalibrated absolute 1e-4. Two native regression tests
(`crates/t0-fast/tests/pool_isolation.rs`, `#[ignore]`d, needs
`bench/ours/` fixtures) load a Q8_0 GGUF model and an F32 safetensors
model against one shared `Engine`, alternate forecast calls A,B,A,B, and
assert each equals its own standalone-model baseline bit-for-bit.

Re-ran the headless compare gate functionally (`node
scripts/headless/run_compare_bench.mjs --url
http://127.0.0.1:8046/bench/compare/ --context 512 --horizon 32 --quant
q8_0`, `pkg-fast` rebuilt from `453e1d1` and synced to `bench/ours/`) --
GPU marker not checked, no timing recorded, machine had another Chrome
process running. All three gates passed:

| comparison | overall max-abs | overall max-abs % range | overall mean-abs | overall mean-abs % range |
|---|---|---|---|---|
| theirs (INT8) vs ours (Q8_0) | 20.07 | 0.883% | 6.46 | 0.284% |
| ours (Q8_0) vs F32 reference | 4.47 | 0.197% | 1.73 | 0.076% |
| theirs (INT8) vs F32 reference | 17.99 | 0.791% | 5.69 | 0.250% |

forecastBatchRows row parity: max-abs 0.001953125 / range 2817.506 =
6.93e-7 relative, well under the 1e-5 gate (same absolute diff as the
original run, now correctly read as float32-rounding noise, not a
failure).

The numbers land exactly where the earlier "72% divergence is
implausible" reasoning predicted: ours-vs-F32 (0.076% mean) is smaller
than theirs-vs-F32 (0.250% mean), consistent with Q8_0 being a milder
quantization than the ONNX INT8 export, and both are now within the same
order of magnitude as theirs-vs-ours (0.284% mean) rather than 72% of
range.

## Performance panel (this run's machine)

- UA: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36`
- GPU adapter (`navigator.gpu.requestAdapter().info`): `apple / metal-3`
- Timestamp: `2026-09-20T17:16:13.948Z`

## Notes

- Prior run's batch-24 caveat (ours replicated one context 24x via
  `forecastBatch`, not a true batch) is resolved this session:
  `forecastBatchRows` gives 24 genuinely distinct shifted windows on both
  engines now. Latency numbers for the fixed batch path are pending the
  cold-machine rerun noted above.
- Follow-ups for a future session, in priority order: (1) fix the
  `Pool`-sharing bug above so the F32-reference agreement rows are
  trustworthy, (2) re-run latency/batch-24 on a cold machine, (3) revisit
  the `forecastBatchRows` parity threshold (absolute 1e-4 -> relative or
  magnitude-scaled) once (1) is fixed, since a Pool fix could also change
  the exact parity number.
