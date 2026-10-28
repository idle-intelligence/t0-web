# Compare page: single-tab head-to-head (onnxruntime-web vs t0-fast)

`bench/compare/index.html` + `bench/compare/compare.js` load both engines
in one tab — theforecastingcompany's `t0-alpha-onnx-int8` export (this run:
revision `1c18a3e2`, 2026-09-13, before its 2026-10-18 re-export) via
`onnxruntime-web` (WebGPU EP, wasm fallback) and our GGUF-resident t0-fast
(the `fast` build of `crates/t0-wasm`, Q8_0 default / Q4_0 button) — so every number is
measured live in whichever browser opens the page, not baked in. The page
also loads a third, F32-residency reference build (`T0Wasm.loadF32`
against the raw `model.safetensors`) to check the two quantized builds
against the unquantized weights, not just against each other. Protocol: 2
warm-ups each, then 10 alternating calls (theirs, ours, …) on identical
input, a 24-signal batch row (24 distinct shifted windows on both
engines, via `forecastBatchRows`), a `forecastBatchRows`-vs-`forecast()`
per-row parity check, and a three-way median/Q10-Q90 agreement diff
(theirs-vs-ours, ours-vs-F32, theirs-vs-F32). Fixture series `us_births`,
context/horizon 512/32, ours at Q8_0.

- Machine: Apple M2. Correctness numbers only; latency was not recorded in
  this run (see below).
- Browser: Playwright's bundled Chromium-for-Testing (`chromium-1243`),
  GPU adapter `apple / metal-3`.
- The page was driven headlessly through its `window.__app.run()` API; that
  harness script is not part of this repository.

## Latency / Batch 24

Not measured in this run: the machine was not idle, so no ms numbers are
recorded. The head-to-head latency rows are in `docs/BENCHMARKS.md`.

## forecastBatchRows row parity

Every one of the 24 distinct-window batch rows, diffed against that same
row's own single-signal `forecast()` call, same Q8_0 model: max-abs diff
0.001953125 births/day, identical across two runs. The first gate (1e-4
absolute) failed on it; it was the wrong bound for data in the thousands,
and the relative gate from `453e1d1` (below) passes.

## Agreement — theirs vs ours

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

The first run's ours-vs-F32 and theirs-vs-F32 rows (72% of range) were
wrong: both models shared one bind-group cache. `89080e8` fixed it; the
corrected rows follow.

## Agreement with the F32 reference, after fixes `89080e8` and `453e1d1`

`89080e8` gives every `GpuModel` its own `Pool` (Engine's device/queue/pipelines stay
shared, its `Pool` field is gone), and `453e1d1` makes the
`forecastBatchRows` parity gate relative (max-abs / range <= 1e-5)
instead of the miscalibrated absolute 1e-4. Two native regression tests
(`crates/t0-fast/tests/pool_isolation.rs`, `#[ignore]`d, needs the
GGUF and safetensors files locally) load a Q8_0 GGUF model and an F32 safetensors
model against one shared `Engine`, alternate forecast calls A,B,A,B, and
assert each equals its own standalone-model baseline bit-for-bit.

Re-ran the compare page headlessly, `pkg-fast` rebuilt from `453e1d1`,
context 512, horizon 32, Q8_0; correctness only, no timing recorded. All
three gates passed:

| comparison | overall max-abs | overall max-abs % range | overall mean-abs | overall mean-abs % range |
|---|---|---|---|---|
| theirs (INT8) vs ours (Q8_0) | 20.07 | 0.883% | 6.46 | 0.284% |
| ours (Q8_0) vs F32 reference | 4.47 | 0.197% | 1.73 | 0.076% |
| theirs (INT8) vs F32 reference | 17.99 | 0.791% | 5.69 | 0.250% |

forecastBatchRows row parity: max-abs 0.001953125 / range 2817.506 =
6.93e-7 relative, well under the 1e-5 gate (same absolute diff as the
original run, now correctly read as float32-rounding noise, not a
failure).

Ours-vs-F32 (0.076% mean) is smaller
than theirs-vs-F32 (0.250% mean), consistent with Q8_0 being a milder
quantization than the ONNX INT8 export, and both are now within the same
order of magnitude as theirs-vs-ours (0.284% mean) rather than 72% of
range.
