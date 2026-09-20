# t0-fast batch GEMM: two register-blocking attempts, both reverted

Machine: Apple M2 (Darwin 25.3.0). Native via `t0-cli bench --backend fast`
(Metal/wgpu-native). GPU-contention rule checked before every timing run
(`pgrep -f 'Chrome for Testing'`, plus this session additionally found and
waited out a sibling Claude session's `cargo run --release --features
native --bin llm-life -- train-vec` job, which is a real GPU consumer not
covered by the task brief's Chrome-only check -- see "contention note"
below). Fixture: `us_births`-equivalent bench window (context 512, horizon
32), weights `theforecastingcompany/t0-alpha` (F32 safetensors, and the
existing `t0-alpha-q8_0.gguf`/`t0-alpha-q4_0.gguf` in `bench/ours/`).
Starting point: commit `e8fae27` (main tip), which already carries five
prior sessions' work (`docs/runs/2026-09-20-perf.md`) ending in a 2x2
register-blocked, 32x32-output-tile, `TK=16` tiled GEMM
(`crates/t0-fast/src/shaders/linear_tiled{,_q8,_q4}.wgsl`).

## Why theirs is faster (this session's contribution to the answer)

Two independent attempts to widen this repo's tiled kernel -- 4x4
register blocking (64x64 output tile) and a deeper K-tile (`TK` 16->32,
same 2x2/32x32 shape) -- both made batch-24 **slower**, not faster, by
4.4x and ~1.45x respectively (measured below, reproducible across
repeats). Both changes increase per-workgroup resource use (accumulator
registers, or shared-memory footprint) without changing total FLOPs; on
this GPU, at this problem's actual size (batch-24 rows are only 384,
`out_dim` a few thousand), that resource increase cuts how many
workgroups can be co-resident per GPU core, and occupancy loss dominates
over the extra arithmetic intensity per shared-memory load. This is the
same failure mode two of the five prior sessions already hit from a
different angle (`tasks_max` and the per-`linear()`-call custom fusion
kernel, both in `docs/runs/2026-09-20-perf.md`) -- "make each dispatch do
proportionally more work" keeps losing to "keep enough small units of
work in flight to hide latency" on this hardware/workload combination.
Read together with that doc's still-unresolved "native is ~3x slower
than the browser's Dawn backend for quantized weights" finding, the
likely honest answer to "why is theirs faster" is that the ONNX Web GPU
EP's matmul either runs through a shader compiled/scheduled by Chromium's
Dawn (which this repo's other sessions already found handles this
workload's dispatch pattern differently, and better, than `wgpu-native`
on the same Metal hardware) or uses a GEMM strategy tuned to Apple GPUs'
actual SIMD-group width (32) and register budget, rather than the
generic "bigger tile, more registers per thread" GEMM-tuning heuristic
this session tried, which is tuned for GPUs with much larger per-thread
register files (desktop NVIDIA/AMD) than Apple's mobile-class GPU
provides. Neither this session nor the five before it had the tooling
budget (a Metal/Dawn shader-compiler trace) to confirm this beyond
inference from the tile-size regression pattern.

## Measurements

All native, `t0-cli bench --backend fast`, warm median (ms/signal for
batch rows, ms for single-signal M=1), same commands each time:
`--context 512 --horizon 32`, single: `--signals 1 --reps 5 --warmup 2`,
batch: `--signals 24 --chunk 24 --reps 3 --warmup 2`.

### Baseline (unchanged from `docs/runs/2026-09-20-perf.md`'s sixth session, re-measured this session for a clean same-machine comparison)

| quant | single M=1 (ms) | batch-24 (ms/signal) |
|---|---|---|
| F32 | 99.3 | 22.7 |
| Q8_0 | 103.2 | 23.2 |
| Q4_0 | 74.4 | 23.3 |

### Attempt 1: 4x4 register blocking (64x64 output tile, still 256 threads/workgroup)

Applied to all three tiled kernels (`linear_tiled{,_q8,_q4}.wgsl`),
`model.rs`'s workgroup-count math changed to `div_ceil(64)`. Parity: all
6 `t0-fast::model` kernel unit tests pass; `t0-cli parity --backend fast`
(short + long) unchanged at 1.251698e-6 / 1.609325e-6 max-abs (F32);
Q8_0/Q4_0-vs-Burn's-own-quantized-path unchanged at 1.490116e-6 /
2.026558e-6 -- bit-identical to baseline, as expected (no numerics
changed, only tile/register shape). Committed (`362d9bf`), then reverted
(`50171be`) once the regression below was confirmed with a paired
same-session A/B (both kernels built and measured back-to-back on the
same otherwise-idle machine state, to rule out the machine-noise swings
this session also hit -- see contention note):

| quant | before (2x2) ms/signal | after (4x4) ms/signal |
|---|---|---|
| F32 (batch-24) | 22.8-22.9 (3 repeats) | 100.1-100.3 (3 repeats) |

Single M=1 untouched by design (M=17-24 < `TILED_THRESHOLD_ROWS=32`
still routes to the pre-existing naive kernel in both variants).

### Attempt 2: deeper K-tile (`TK` 16 -> 32, F32 kernel only, 2x2/32x32 shape kept)

Doubles the K-loop's shared-memory tile depth (halving barrier-
synchronized loop iterations, doubling shared-memory footprint per
workgroup from 4KB to 8KB). Kernel unit test (`f32_linear_tiled`) still
passes at float32-rounding tolerance. Not committed (measured, reverted
immediately once regression was clear):

| quant | before (`TK=16`) ms/signal | after (`TK=32`) ms/signal |
|---|---|---|
| F32 (batch-24) | 22.7 | 33.0-33.6 (2 repeats) |

## What didn't help (and why, per the numbers above)

- **4x4 register blocking**: quartered workgroup count (fewer independent
  units of work to hide latency) and quadrupled register pressure per
  thread (16 accumulators + 8 more for the per-k-step load registers) --
  a 4.4x regression, the clearest signal this session produced.
- **Deeper K-tile (`TK=32`)**: doubled shared-memory footprint per
  workgroup without changing FLOPs or workgroup count -- still a clear
  ~1.45x regression, smaller than the 4x4 attempt but the same mechanism
  (per-workgroup resource pressure cutting occupancy).
- Both point the same direction: this workload's batch-24 rows (384) and
  weight dimensions are small enough, and Apple's mobile-class GPU has
  few enough concurrently-resident-workgroup slots per core, that the
  currently-shipped 2x2/32x32/`TK=16` kernel is already closer to this
  GPU's actual occupancy sweet spot than either "obvious" GEMM-tuning
  move (bigger register tile, deeper K-tile) that generic (desktop-GPU-
  oriented) matmul tuning guides recommend next.

## Not attempted this session

`dot4I8Packed` (task brief's optional item 3): skipped given the above --
two independent attempts at "more compute per dispatch" both regressed on
this hardware/workload, and `dot4I8Packed` is the same category of change
(more arithmetic intensity per instruction, not more concurrent work), so
it carries the same regression risk without new evidence it would land
differently. Fused bias+residual epilogue (task brief's item 2 fusion
note): not attempted -- `residual_block`/`mhsa`/`swiglu_ffn`'s
`add_inplace` calls after `linear()` would need a new residual-buffer
binding threaded through both `LinearDims`/`LinearQDims` and every
pipeline's bind-group layout, a larger and riskier change than this
session's budget allowed once the two GEMM-tiling attempts above had
already used the session on a negative result; flagged as the next
concrete lever (each of `add1`/`add2`/residual-block's `add` is a whole
extra dispatch per occurrence -- 2/layer x 24 layers -- that a fused
epilogue would remove).

## Net effect of this session

No code changed in the final state: both attempted kernel changes were
measured and reverted; `git diff e8fae27 -- crates/t0-fast` is empty.
Parity gate unchanged (1.25e-6/1.61e-6 F32 short/long, 1.49e-6/2.03e-6
Q8_0/Q4_0-vs-Burn). `cargo clippy -p t0-fast -p t0-cli -p t0-core
--release --no-default-features --features fast,ndarray -- -D warnings`
clean; `wasm-pack build crates/t0-wasm --target web --out-dir pkg-fast
--release --no-default-features --features fast` builds clean (not
copied anywhere, per the task brief). Batch-24 remains at this repo's
prior best (~11.2ms/signal in the browser per
`docs/runs/2026-09-20-perf.md`'s sixth session, ~22.7-23.3ms/signal
native -- the native/browser gap itself is the open question that doc
already flagged and this session's regressions are consistent with, not
new evidence toward resolving).

## Contention note (process, not a result)

This session hit two separate sources of shared-M2-GPU contention beyond
the task brief's named check: (1) `pgrep -f 'Chrome for Testing'`
non-zero for a sustained period at the session's start (another job's
browser benchmark), waited out via a background poll before any timing
run; (2) a sibling Claude session (`llm-life`, `cargo run --release
--features native --bin llm-life -- train-vec`) actively compiling/
running mid-session, discovered only because a batch-24 measurement came
back at 49.6ms/signal -- more than double the otherwise-reproducible
22.7ms/signal baseline -- prompting a process-table check rather than
accepting the number. Waited out the same way. All numbers reported above
are from windows where `pgrep` for both `Chrome for Testing` and
`llm-life`/`cargo run` returned zero, each confirmed immediately before
the measurement it's paired with.
