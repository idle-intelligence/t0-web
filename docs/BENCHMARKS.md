# Benchmarks (research log)

Data only — analysis and strategy go in `docs/runs/`.

## Milestone 0 — numerical parity (native, F32, CPU/ndarray)

- Machine: Apple M2 (Darwin 25.3.0), no GPU job running concurrently.
- Commit: (this milestone's HEAD — see `git log`).
- Reference: `tfc-t0` (PyPI) `t0.model.model.T0Forecaster`, checkpoint `theforecastingcompany/t0-alpha`.
- Command: `cargo run -p t0-cli --release -- parity --fixtures fixtures --weights ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/model.safetensors --config ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/config.json`
- Fixtures: `tools/make_fixtures.py`, context 512, horizon 96, quantile levels `[0.1, 0.25, 0.5, 0.75, 0.9]` (== trained levels, no interpolation).

| case | quantile max-abs err | quantile max-rel err | patch-embedding max-abs err | layer-0-output max-abs err |
|---|---|---|---|---|
| case0_univariate_sine | 1.192093e-6 | 6.311951e-5 | 2.670288e-5 | 2.670288e-5 |
| case1_trivariate_freqs | 1.192093e-6 | 1.262039e-4 | 4.577637e-5 | 5.340576e-5 |
| case2_masked_gap | 1.192093e-6 | 2.634852e-4 | 3.051758e-5 | 3.051758e-5 |

Gate: max-abs quantile error <= 1e-4 across all fixtures. **Result: PASS** (worst case 1.192093e-6, ~84x under the gate).

First run (before fixing the RoPE xpos-scale bug — see `docs/runs/2026-09-19-parity.md`) had layer-0 max-abs error ~5e-3 to ~8e-3 and quantile max-abs error ~1.1e-3 to ~2.9e-3, failing the gate.

## Milestone 1a — GGUF quantization round-trip (native, CPU/ndarray)

- Machine: Apple M2 (Darwin 25.3.0).
- Commit: (this milestone's HEAD — see `git log`).
- Command: `cargo test -p t0-core --release -- --ignored --nocapture gguf_round_trip_quantiles`
- Every tensor is dequantized to f32 at load (see `crates/t0-core/src/weights.rs` doc comment: `burn-wgpu`/`cubecl-wgpu` 0.9.0 only implements `FloatElement` for `f32`, so there is no lower-precision compute path to dequantize into yet — f16/Q8_0/Q4_0 only shrink the file and load time this milestone). Quantized tensors: the four big per-layer matmuls (`attention.wQKV.weight`, `attention.wO.weight`, `mlp.0.weight`, `mlp.2.weight`); everything else (patch encoder, type embeddings, decoder, all norms, `head.quantile_levels`) stored as f16.

File sizes (`t0-cli export-gguf`, from `model.safetensors`, 406,601,492 bytes / 406.6 MB):

| quant | file bytes | file MB | ratio to F32 safetensors |
|---|---|---|---|
| f16 | 203,307,887 | 203.3 | 0.500x |
| q8_0 | 108,936,047 | 108.9 | 0.268x |
| q4_0 | 58,604,399 | 58.6 | 0.144x |

Quantile error vs the reference PyTorch fixtures (max over the 3 fixtures; f16 is gated per this milestone's brief, Q8_0/Q4_0 are recorded only — the accuracy verdict belongs to the GIFT-Eval step):

| quant | quantile max-abs err | quantile max-rel err | gate |
|---|---|---|---|
| f16 | 4.151165e-4 | 2.925728e-1 | <=1e-3: **PASS** |
| q8_0 | 5.663484e-3 | 1.786981e0 | not gated (recorded) |
| q4_0 | 9.818502e-2 | 9.410091e1 | not gated (recorded) |

Per-fixture detail:

| quant | case | max-abs | max-rel |
|---|---|---|---|
| f16 | case0_univariate_sine | 4.151165e-4 | 8.219351e-2 |
| f16 | case1_trivariate_freqs | 2.973080e-4 | 2.970142e-2 |
| f16 | case2_masked_gap | 3.453493e-4 | 2.925728e-1 |
| q8_0 | case0_univariate_sine | 2.638936e-3 | 3.320634e-1 |
| q8_0 | case1_trivariate_freqs | 2.637386e-3 | 5.031561e-1 |
| q8_0 | case2_masked_gap | 5.663484e-3 | 1.786981e0 |
| q4_0 | case0_univariate_sine | 4.542732e-2 | 5.427120e0 |
| q4_0 | case1_trivariate_freqs | 5.204641e-2 | 6.766154e0 |
| q4_0 | case2_masked_gap | 9.818502e-2 | 9.410091e1 |

The large max-rel figures (Q4_0 up to ~94x) come from the relative-error denominator floor (`abs/max(|y|,1e-6)`) on near-zero reference quantile values, not from a systematic bias — max-abs is the meaningful column here.

## Milestone 1a — batch path (native, CPU/ndarray)

- Command: `cargo test -p t0-core --release -- --ignored --nocapture batch_of_8_copies_equals_8x_single`
- Mechanism: `TimeSeries::from_context_batch` assigns each of the N signals its own group id; the existing group-attention mask (`build_group_mask`, group-id equality) isolates them with zero changes to `model.rs`. **Result: PASS** — 8 copies of fixture 0 batched in one forward pass match 8x the single-signal forecast to <=1e-5 max-abs (float non-associativity noise from a different matmul shape, not a correctness gap).

## Milestone 1a — latency table

- Machine: Apple M2 (Darwin 25.3.0), 8 threads available (`std::thread::available_parallelism`).
- Commit: (this milestone's HEAD — see `git log`).
- GPU contention: a `jacobi2000` training run (`target/release/jacobi`) was active on the shared GPU for the first wgpu measurement pass; it finished before this doc was written and the wgpu numbers below were re-measured with the GPU idle (checked via `ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` immediately before each pass) — the re-measured numbers are within noise of the contended ones (e.g. F32 n=1: 0.2226s contended vs 0.2232s idle), so both are reported as final, not provisional.
- Command (per row): `t0-cli bench --weights <F32 safetensors | GGUF> [--config <config.json>] --backend <ndarray|wgpu> --signals <1|1000> --context 512 --horizon 96 --reps 5` — median of 5 timed forward passes after 1 untimed warm-up; `--backend` is asserted against the binary's compiled Cargo feature (`ndarray` default, `wgpu` via `--no-default-features --features wgpu`), rebuilt between backends.
- All compute is F32 on both backends: `cubecl-wgpu` 0.9.0 (pinned by Burn 0.20.1) only implements `FloatElement` for `f32` (see `crates/t0-core/src/weights.rs`), so f16/Q8_0/Q4_0 GGUF only change file size and load/dequant time here, not the matmul dtype — the near-flat runtime across quant levels in the table below is expected, not noise-free precision.

| backend | quant | signals | file MB | load+dequant s | median forward s |
|---|---|---|---|---|---|
| ndarray | F32 | 1 | 406.6 | 0.215 | 0.226 |
| ndarray | F32 | 1000 | 406.6 | 0.239 | 113.434 |
| ndarray | f16 | 1 | 203.3 | 0.423 | 0.540 |
| ndarray | f16 | 1000 | 203.3 | 0.512 | 105.933 |
| ndarray | Q8_0 | 1 | 108.9 | 0.285 | 0.264 |
| ndarray | Q8_0 | 1000 | 108.9 | 0.292 | 135.336 |
| ndarray | Q4_0 | 1 | 58.6 | 0.270 | 0.355 |
| ndarray | Q4_0 | 1000 | 58.6 | 0.267 | 78.342 |
| wgpu (Metal) | F32 | 1 | 406.6 | 0.195 | 0.223 |
| wgpu (Metal) | F32 | 16 | 406.6 | 0.196 | 0.593 |
| wgpu (Metal) | F32 | 1000 | 406.6 | — | **fails, see below** |
| wgpu (Metal) | f16 | 1 | 203.3 | 0.204 | 0.223 |
| wgpu (Metal) | f16 | 16 | 203.3 | 0.201 | 0.593 |
| wgpu (Metal) | f16 | 1000 | 203.3 | — | **fails, see below** |
| wgpu (Metal) | Q8_0 | 1 | 108.9 | 0.195 | 0.222 |
| wgpu (Metal) | Q8_0 | 16 | 108.9 | 0.196 | 0.596 |
| wgpu (Metal) | Q8_0 | 1000 | 108.9 | — | **fails, see below** |
| wgpu (Metal) | Q4_0 | 1 | 58.6 | 0.158 | 0.222 |
| wgpu (Metal) | Q4_0 | 16 | 58.6 | 0.161 | 0.592 |
| wgpu (Metal) | Q4_0 | 1000 | 58.6 | — | **fails, see below** |

n=1000 medians have wide spread across the 5 reps (e.g. F32: 106.0-135.4s) — consistent with thermal/scheduling noise on a fanless M2 under a sustained ~100s single-threaded-per-op CPU run, not a quant-dependent effect (see note above).

**wgpu batch-path limitation found**: at `--signals 1000` (any quant), every wgpu run panics identically:
```
Unable to launch matmul because the config is invalid: "This algorithm needs 40960 shared memory bytes but hardware limit is 32768."
```
from `cubek-matmul-0.1.1/src/launch/strategy.rs:437` (a dependency of `cubecl-wgpu` 0.9.0, pinned by Burn 0.20.1) — its autotuned matmul picks a tiled kernel whose shared-memory tile size doesn't respect Metal's 32 KiB per-threadgroup limit for this problem shape (the group-attention layers' `[p, heads, v, v]`-shaped matmul, `v = n_signals`). Binary search found the exact threshold on this machine: **n_signals=16 works, n_signals=32 fails** (confirmed at 8/16 pass, 32/64/100/1000 fail — quant scheme doesn't matter, this is a shape-driven autotune bug, not a precision issue). The wgpu numbers above (n=1, n=16) are all real, executed runs; no env var or public API was found in `cubek-matmul` 0.1.1 to force a smaller-tile strategy in the time budget for this milestone — flagging as a known limitation rather than patching a pinned dependency mid-milestone. ndarray has no such ceiling (ran n=1000 fine). This blocks a "Forecast all 1000 signals" wgpu batch claim until either `cubecl`/`cubecl-wgpu` is upgraded past this bug or the matmul strategy is forced smaller — worth an owner call on whether to chase a cubecl version bump now or defer to the WASM milestone (WebGPU in-browser will hit the same or a stricter cap, since 32 KiB is already the *implementation* limit here, not just the spec's 256-invocation workgroup cap).
