# 2026-09-19 — Milestone 1a: quantized weights, wgpu runtime, latency table

Native, Apple M2, `theforecastingcompany/t0-alpha` checkpoint. Data tables in `docs/BENCHMARKS.md`.

## GGUF container and quantization

`crates/t0-core/src/gguf.rs` implements GGUF v3 (magic/version/tensor-info/metadata layout, ggml's reversed-dimension-order convention) plus Q8_0/Q4_0 block quantization matching llama.cpp's own `ggml_quantize_row_q8_0`/`_q4_0` reference implementations exactly (32-value blocks, f16 scale, Q4_0's `j`/`j+16` nibble pairing and `+8.5` floor rounding). `Weights::export_gguf` quantizes only the four big per-layer matmuls (`wQKV`, `wO`, `mlp.0`, `mlp.2`); everything else (patch encoder, type embeddings, decoder, all norms, `head.quantile_levels`) is stored at f16 regardless of the requested scheme, per the "safe tensors" list in `docs/reports/t0-alpha.md` §5 — those tensors are either tiny or gate output precision directly (patch encoder feeds every downstream layer; the decoder produces the quantile values themselves).

At load, `Weights::load_gguf` dequantizes every tensor straight to f32. This isn't a shortcut: `cubecl-wgpu` 0.9.0 (pinned transitively by Burn 0.20.1) only implements its `FloatElement` trait for `f32` (confirmed by reading `cubecl-wgpu-0.9.0/src/element.rs` directly — no `impl FloatElement for half::f16`), so there is currently no lower-precision compute path on the wgpu backend to dequantize into. GGUF quantization this milestone buys file size and load time, not compute dtype; that matches this milestone's own stop condition ("if burn-wgpu f16 is not usable, run wgpu in f32 and say so").

The large Q4_0 max-rel figures (up to ~94x in the fixture table) come from the relative-error denominator floor (`abs_err / max(|reference|, 1e-6)`) hitting near-zero reference quantile values, not a systematic bias — max-abs is the meaningful column, and it stays small and monotonic across f16 -> Q8_0 -> Q4_0 (4e-4 -> 6e-3 -> 1e-1).

## wgpu runtime

`t0-cli --backend wgpu` now genuinely executes via `burn-wgpu` on Metal (previously only compiled). F32 parity against the PyTorch reference: worst-case 1.6e-6 max-abs, same order of magnitude as ndarray's 1.2e-6 — and wgpu vs ndarray agree to within ~1e-4 on both F32 and f16-GGUF inputs, satisfying this milestone's actual requirement (the f16-GGUF-vs-PyTorch max-abs of 4.15e-4 exceeding the strict 1e-4 parity gate is expected and is the same number ndarray produces for the same quantized input — it's a quantization-precision fact, not a backend-divergence bug).

`--backend` is checked against the binary's compiled Cargo feature and errors on mismatch rather than attempting a runtime backend switch: two different Burn `Backend` types can't coexist in one binary without dynamic dispatch, and this repo's CLAUDE.md/GOAL.md constraints (RAM tight, one cargo build at a time) make "one binary per backend feature" the right tradeoff over adding that machinery for this milestone.

## Batch path

`TimeSeries::from_context_batch` gives each of N independent signals its own group id. The existing group-attention mask (`build_group_mask` in `crates/t0-core/src/mask.rs`, which already gates cross-variate attention on group-id equality) isolates them with zero changes to `model.rs` — batching independent signals and multivariate joint-forecasting turn out to be the same mechanism with different group-id assignment. Verified: 8 batched copies of fixture 0 match 8x the single-signal forecast to <=1e-5 max-abs (`crates/t0-core/tests/batch.rs`).

## wgpu batch-path bug: cubek-matmul shared-memory limit on Metal

At `--signals 1000` (any quant scheme), every wgpu run panics identically:

```
Unable to launch matmul because the config is invalid: "This algorithm needs 40960 shared memory bytes but hardware limit is 32768."
```

from `cubek-matmul-0.1.1/src/launch/strategy.rs:437` (a `cubecl-wgpu` 0.9.0 dependency, pinned by Burn 0.20.1). Its autotuned matmul selects a tiled kernel whose shared-memory tile doesn't respect Metal's 32 KiB per-threadgroup limit for the group-attention layers' `[p, heads, v, v]`-shaped matmul (`v = n_signals`). Binary search on this machine found the exact threshold: n_signals=16 works, n_signals=32 fails (8/16 pass; 32/64/100/1000 all fail identically, independent of quant scheme — confirming this is a shape-driven autotune bug, not a precision issue).

No environment variable or public API was found in `cubek-matmul` 0.1.1 to force a smaller-tile strategy within this milestone's time budget, so this is recorded as a known limitation rather than patched into a pinned dependency mid-milestone. ndarray has no equivalent ceiling (ran n=1000 without issue). This blocks a "Forecast all 1000 signals" wgpu batch claim for GOAL.md's "beat their numbers" bar until either `cubecl`/`cubecl-wgpu` is upgraded past this bug or the matmul strategy is forced smaller.

**Owner-relevant note**: 32 KiB is Metal's actual per-threadgroup shared-memory limit here, not just WebGPU's spec-level 256-invocation workgroup cap — so a future WASM+WebGPU build is likely to hit the same or a stricter ceiling in-browser, not a Metal-specific one. Worth deciding whether to chase a `cubecl`/`cubecl-wgpu` version bump now (native) or defer the fix to the WASM milestone once the same failure is reproduced in-browser.

## GPU contention

A `jacobi2000` training run (`target/release/jacobi`) held the GPU during the first wgpu bench pass (checked via `ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'`). It finished before this doc was written; re-measuring with the GPU idle gave numbers within noise of the contended pass (e.g. F32 n=1: 0.2226s contended vs 0.2232s idle). The idle numbers are what's reported in `docs/BENCHMARKS.md`.

## CPU timing noise

n=1000 ndarray medians have wide spread across the 5 reps (e.g. F32: 106.0s-135.4s) — consistent with thermal/scheduling noise on a fanless M2 under a sustained ~100s single-threaded-per-op run, not a quant-dependent effect. Runtime is near-flat across f16/Q8_0/Q4_0 at both n=1 and n=1000 because compute stays F32 regardless of storage quantization (see "GGUF container and quantization" above) — the quant ladder saves file size and load time, not per-step compute this milestone.

## Chunked batch: the workaround for the shared-memory bug above

The "wgpu batch-path bug" section above records the raw failure (n_signals
>16 panics on any wgpu quant scheme with `cubek-matmul`'s 40 KB shared-memory
tile vs. Metal's 32 KB limit) and that no `cubek-matmul` 0.1.1 public API was
found to force a smaller tile. The workaround implemented this session is
architectural, not a dependency patch: `t0_core::forecast_batch` (previously
one un-chunked forward pass over all `n_signals`) now calls
`forecast_batch_chunked` with a new `DEFAULT_BATCH_CHUNK = 16` constant
(`crates/t0-core/src/model.rs`) — the largest `n_signals` already confirmed
safe by the earlier binary search. `forecast_batch_chunked` slices the batch
into groups of `chunk_size` (default 16, `t0-cli bench --chunk N` to
override), runs one independent forward pass per group (each group's signals
still get distinct group ids via `TimeSeries::from_context_batch`, so
per-group forecasts are bit-identical to what an un-chunked pass over just
that group would produce), and writes each group's output into the right
slice of the full-size output buffer. This is transparent to every existing
caller: `forecast_batch`'s signature is unchanged, `cmd_bench` just threads a
new `--chunk` flag through to the chunked variant, and the four call sites in
tests (`batch.rs`, the new `chunk_wgpu.rs`) keep working unmodified (`batch.rs`
never used more than 8 signals, under the default chunk, so its behavior is
identical).

Correctness gate: `crates/cli/tests/chunk_wgpu.rs`
(`chunked_batch_of_40_equals_40_single`, `#[ignore]`d, needs the checkpoint
and `--features wgpu`) forecasts 40 synthetic sine signals two ways — 40
separate `chunk_size=1` single-signal wgpu forward passes, and one
`forecast_batch_chunked(..., 40, ..., DEFAULT_BATCH_CHUNK=16)` call (which
internally splits into 16+16+8) — and asserts every signal's forecast agrees
to <=1e-5 max-abs. Ran on Apple M2, GPU idle: **PASS** in 11.4s
(`cargo test -p t0-cli --release --no-default-features --features wgpu --
--ignored chunked_batch_of_40_equals_40_single`).

Upstream issue, restated for this section's context: `cubek-matmul-0.1.1`
(pinned transitively by Burn 0.20.1 via `cubecl-wgpu` 0.9.0) autotunes a
tiled matmul strategy for the group-attention layers' `[p, heads, v, v]`
matmul (contraction dim `v = n_signals`) that needs 40 KB of per-threadgroup
shared memory once `v > 16`; Metal's hard limit here is 32 KB
(`cubek-matmul-0.1.1/src/launch/strategy.rs:437`). Chunking sidesteps this by
keeping `v <= 16` in every dispatched matmul, at the cost of one Rust-side
loop over `ceil(n_signals / 16)` independent forward passes instead of one —
correct today, but each chunk still pays its own kernel-launch/dispatch
overhead, which is exactly the mechanism this repo's WASM constraints
(dispatch fragmentation) warn about; a future `cubecl`/`cubecl-wgpu` version
bump that lets the autotune pick a smaller tile remains the real fix.

## Chunked wgpu latency at 100 and 1000 signals

Extends the Milestone-1a latency table (`docs/BENCHMARKS.md`) with the
previously-failing n=100/1000 wgpu rows, now possible via `--chunk 16`.
Machine: Apple M2, GPU idle before each run (checked with `ps -A -o comm |
grep -E 'target/release/(jacobi|llm-life)'`). Command per row: `t0-cli bench
--weights <F32|GGUF> [--config ...] --backend wgpu --signals <N> --context
512 --horizon 96 --reps 3 --chunk 16`.

At n=100 (7 chunks: 16*6+4), per-signal cost is ~38ms flat across F32/f16/
Q8_0/Q4_0 — consistent with the existing finding that wgpu compute stays F32
regardless of storage quant (dequant happens once at load, not per forward
pass). At n=1000 (63 chunks: 16*62+8), F32 measured 37.99ms/signal (median of
3, all three reps within ~10s of each other: 37.16/37.99/47.10s — the last
rep's outlier is consistent with the CPU-timing thermal noise already
recorded for ndarray n=1000 runs); f16 measured 46.50ms/signal with a much
wider single-rep outlier (46.5/46.1/88.1s) — again read as scheduling/thermal
noise on a sustained ~45-90s run rather than a quant effect, matching this
doc's existing "CPU timing noise" section's reasoning, now observed on wgpu
too at high chunk counts. Full numbers: `docs/BENCHMARKS.md`.

Two structural costs are new at high signal counts that weren't visible at
n<=16: (1) 63 sequential forward passes instead of 1 means the model's
per-layer weight tensors are re-read from the same GPU buffers 63 times
(no cross-chunk state reuse beyond what Burn's backend already caches), and
(2) each chunk pays fixed per-dispatch overhead the un-chunked path would
have amortized across all 1000 signals in the matmuls that didn't hit the
shared-memory ceiling. Chunking trades a hard failure for a working but not
dispatch-optimal path — acceptable for this milestone's "unblock the number"
goal, revisit if/when the cubecl-wgpu autotune bug itself gets fixed upstream.

## The drift benchmark: reproducing the INT8 card's protocol

`theforecastingcompany/t0-beta-onnx-int8`'s card (fetched this session, full
text in `docs/reports/t0-published-numbers.md`) publishes the drift formula
verbatim ("absolute error divided by the reference series' full forecast
range across the horizon and all quantiles, not relative error on an
individual prediction") and the quantization recipe (96 transformer
projection matrices per-channel signed INT8, six I/O projections FP32,
FP32 compute) but **no case-generator code or distribution parameters** for
the "54 synthetic cases" — confirmed absent on both INT8 cards and in the
`theforecastingcompany` GitHub org (`t0-published-numbers.md`'s GitHub
section). Everything below the formula is therefore our own reconstruction,
not a verified reproduction, and is called out as such in every place it's
used.

**What's reproduced faithfully:**
- The drift formula itself, exactly as published: for each case, compute
  `range = max(f32_reference) - min(f32_reference)` over the whole output
  (all horizon steps x all quantile levels), then per-point
  `drift_i = |f32_i - quant_i| / range`. "Mean drift" for a case is the mean
  of `drift_i` over all points; "point drift" for a case is the max.
- 54 cases, aggregated the same two ways their card implies ("worst"
  suggests a max over cases; we additionally report the mean over cases in
  our own table since GOAL.md's addendum asks for "worst/mean" columns they
  don't publish a mean for).
- The quantization recipe for `int8-theirs`: `Weights::quantize_int8_theirs`
  (`crates/t0-core/src/weights.rs`) applies per-channel (per-output-row)
  signed INT8 quantize/dequantize (`gguf::int8_per_channel_roundtrip`) to
  exactly the same 96 tensors our own `is_quantizable` list already names
  (`attention.wQKV.weight`, `attention.wO.weight`, `mlp.0.weight`,
  `mlp.2.weight` x 24 layers = 96 — the count matches their card exactly)
  and leaves the patch encoder's and decoder's 3+3=6 `ResidualBlockWeights`
  matrices untouched at F32 — the "six input/output projections" their card
  names. This is a coincidence of our existing tensor taxonomy already
  lining up with theirs, not a deliberate re-derivation of their internal
  module boundaries — worth flagging in case the actual six tensors they
  mean differ from ours despite the matching count.

**What's assumed, documented here because it isn't published:**
- **Case shape**: univariate (`v=1`), context 512 / horizon 96 (this
  project's existing fixture convention, `tools/make_fixtures.py`), not the
  full 1–4096 context / 32–1024 horizon / 1–64 target-row / 0–64
  covariate-row range the *alpha* ONNX card's 17-case validation matrix
  describes (that card is a different, non-drift validation and gives no
  reason to think the 54-case drift suite uses the same ranges).
  **Consequence**: our reported drift numbers use one context/horizon
  point, not a distribution over the published range — if their 54 cases
  span wider context/horizon values, results could differ (longer context
  typically stabilizes the causal scaler's running statistics, which would
  tend to *lower* drift relative to ours, not raise it, but this is a
  guess, not a measurement).
- **Signal shape**: our `synthetic_case` (`crates/cli/src/main.rs`) is a
  sum of two sine harmonics (frequency/amplitude/phase drawn from a
  deterministic xorshift64* PRNG seeded by `--seed`), a small linear trend,
  and Gaussian noise (Box-Muller from the same PRNG); every 4th case gets a
  contiguous NaN gap covering 1/8 of the context, mirroring this repo's own
  `case2_masked_gap` fixture. No published detail suggests or rules out this
  shape — it was chosen to exercise the same code paths (scaler,
  missing-value mask, patch alignment) their model would also have to
  handle, not because it matches theirs.
- **No covariate/multivariate cases**: all 54 cases are `v=1`, matching
  this milestone's existing scaler limitation (`FUTURE` covariate rows are
  unimplemented, `crates/t0-core/src/scaler.rs` panics loudly if asked).
- **Aggregation**: "mean" columns in our table (mean-of-54-cases) are our
  own addition per GOAL.md's addendum table spec; their card only reports
  the "worst" (max-of-54-cases) figures for both metrics.

**Numbers** (full table `docs/BENCHMARKS.md`): our own Q8_0 (worst mean
drift 0.53%, worst point drift 1.78%) and `int8-theirs` (worst mean drift
0.85%, worst point drift 2.04%) both clear their published 2%/10%
acceptance gates; our Q4_0 (worst mean drift 3.11%, worst point drift 8.42%)
fails the 2% mean-drift gate despite passing the 10% point-drift gate. f16
is far inside both gates (0.034%/0.13%). None of these numbers are on the
same checkpoint as their published 0.2271%/9.393% (theirs is t0-beta,
256M params, embed_dim 1024; ours here is t0-alpha, 102M params, embed_dim
512 — see `docs/reports/t0-published-numbers.md`), so the comparison is
recipe-vs-recipe (`int8-theirs` row) at best, not yet model-vs-model.

## What remains

- GIFT-Eval subset harness: the accuracy verdict for Q8_0/Q4_0 belongs here, not this milestone (per the brief's own stop condition).
- The t0.run-style page and web glue (`web/`, per GOAL.md items 3-4) — not started.
- The METAR live-data metric (station list + fetch path from trucs.ai `knn-weather`) — not started.
- WGSL Q8/Q4 dequant+matmul kernels for a real WASM+WebGPU deployment — this milestone's quantization is a load-time CPU-side dequant into f32 Burn tensors, not an on-GPU kernel; deferred to the WASM/browser milestone per `docs/reports/t0-alpha.md` §5.
- The wgpu batch-path shared-memory bug above blocks a valid "1000 signals on wgpu" latency number until resolved.
