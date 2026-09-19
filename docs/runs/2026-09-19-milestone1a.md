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

## What remains

- GIFT-Eval subset harness: the accuracy verdict for Q8_0/Q4_0 belongs here, not this milestone (per the brief's own stop condition).
- The t0.run-style page and web glue (`web/`, per GOAL.md items 3-4) — not started.
- The METAR live-data metric (station list + fetch path from trucs.ai `knn-weather`) — not started.
- WGSL Q8/Q4 dequant+matmul kernels for a real WASM+WebGPU deployment — this milestone's quantization is a load-time CPU-side dequant into f32 Burn tensors, not an on-GPU kernel; deferred to the WASM/browser milestone per `docs/reports/t0-alpha.md` §5.
- The wgpu batch-path shared-memory bug above blocks a valid "1000 signals on wgpu" latency number until resolved.
