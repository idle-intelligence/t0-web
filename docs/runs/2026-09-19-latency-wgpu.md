# Single-signal native wgpu (Metal) latency, per quant

- Machine: Apple M2 (Darwin 25.3.0), `wgpu` backend (`burn-wgpu`, Metal).
  GPU rule checked before every run: `pgrep -fl 'llm-life train-a|eval-metrics'`
  empty (confirmed empty for the whole session covered by this doc). No
  other local `cargo`/`rustc` build was running while these numbers were
  taken (one remote build over `ssh` to a different box, not competing for
  local CPU/GPU).
- Commit: this doc's commit, `t0-cli` built with
  `cargo build -p t0-cli --release --no-default-features --features wgpu`.
- Weights: `../models/hf/theforecastingcompany/t0-alpha/{model.safetensors,config.json}`
  (F32); `/tmp/t0-alpha-{f16,q8_0,q4_0}.gguf` (config embedded, from this
  session's earlier `t0-cli export-gguf`, same files as
  `docs/runs/2026-09-19-latency-cpu.md`).
- Command, cold (first call, includes autotune/shader compile):
  `t0-cli bench --weights <F32|GGUF> [--config config.json] --backend wgpu --signals 1 --context 512 --horizon 32 --warmup 0 --reps 1`
- Command, warm (median of 10 after 2 untimed warm-ups, same protocol as
  the CPU table):
  `t0-cli bench --weights <F32|GGUF> [--config config.json] --backend wgpu --signals 1 --context 512 --horizon 32 --warmup 2 --reps 10`

| quant | file MB | load time (s) | cold first-call (ms) | warm median forward latency (ms) |
|---|---|---|---|---|
| F32 | 406.6 | 0.232-0.340 | 1031.9 | 945.6 |
| f16 (load-time cast) | 203.3 | 0.265-0.295 | 854.6 | 883.9 |
| Q8_0 | 108.9 | 0.376-1.453 | 1262.4 | 746.4 |
| Q4_0 | 58.6 | 1.075-1.195 | 1657.2 | 704.6 |

Raw warm reps (all 10, seconds) per row:

- F32: `[1.018000958, 1.02112525, 0.811056875, 0.865566709, 0.836932292, 0.973995625, 0.945582959, 0.955895709, 0.929806417, 0.888444959]`
- f16: `[0.790240375, 0.757692125, 0.990135958, 1.0251305, 1.049516917, 0.998139416, 0.883935125, 0.822786792, 0.848797458, 0.803758791]`
- Q8_0: `[0.770642584, 0.735540875, 0.725867416, 0.6902765, 0.717319167, 0.78356175, 0.618066666, 0.749210834, 0.778199166, 0.746356833]`
- Q4_0: `[0.718384583, 0.741696417, 0.695471541, 0.6993405, 0.704583292, 0.750596292, 0.702973875, 0.682140333, 0.706111208, 0.697008542]`

## Reading the numbers

- **Cold vs warm**: cold first-call latency (1032-1657 ms) is higher than
  the warm median (705-946 ms) for every quant except f16, where cold
  (854.6 ms) happens to land inside the warm rep spread (758-1050 ms) —
  read as run-to-run noise on a single untimed sample, not a real
  "f16 has no compile cost" finding. The cold/warm gap for Q8_0 (1262 vs
  746 ms, +69%) and Q4_0 (1657 vs 705 ms, +135%) is the clearest signal:
  smaller quants have a *larger* relative one-time cost here, consistent
  with `cubecl`'s autotune needing to pick a matmul/dequant kernel variant
  the first time a given tensor shape+dtype combination runs, and Q4_0's
  extra dequant-to-F32 step at load/first-use.
- **Warm latency does not scale down with smaller quant** (F32 945.6, f16
  883.9, Q8_0 746.4, Q4_0 704.6 ms) — same fact already established for
  the CPU table and the chunked-batch wgpu table in `docs/BENCHMARKS.md`:
  compute stays F32 regardless of storage quant (dequant happens once at
  load), so quant only buys file size and load time, not forward-pass
  speed, on this backend either. The mild downward trend (F32 > f16 >
  Q8_0 ~ Q4_0) is within the same noise band as the per-rep spread (e.g.
  Q8_0's own reps span 618-784 ms) and should not be read as a real
  quant-dependent speedup.
- **wgpu (Metal) vs native ndarray (CPU)**, same quant, same shape (context
  512, horizon 32, 1 signal): wgpu's warm median (705-946 ms) is **slower**
  than ndarray's (363-424 ms, `docs/runs/2026-09-19-latency-cpu.md`) at
  this batch size. Expected and consistent with this repo's existing
  finding (`docs/BENCHMARKS.md`'s Milestone-1a latency table): at
  `n_signals=1` the fixed per-dispatch overhead of many small GPU kernel
  launches (one matmul/attention op at a time, no batching) dominates over
  the actual compute, so a single CPU thread doing the same tiny matmuls
  in one process is faster; wgpu only wins once batched (see the
  `docs/BENCHMARKS.md` "chunked wgpu batch" table, ~38-52 ms/signal at
  n=100-1000).
- **Compute class**: this table's compute path is F32 throughout (all four
  quants dequant to F32 at load, matmuls run in F32) — the same class as
  the published `t0-alpha-onnx-int8` card's "INT8-weight, FP32-compute"
  (`docs/reports/t0-published-numbers.md`), so once their browser timing is
  available this is an apples-to-apples row on compute precision.

## What this doesn't cover

Batched (n>1) wgpu latency at ctx=512/horizon=32 (only the n=1 single-signal
protocol was run here, to match the CPU latency table directly) — the
existing `docs/BENCHMARKS.md` chunked-batch table already covers batched
wgpu at horizon=96, a different shape, not repeated here. WebGPU-in-browser
latency: see `docs/runs/2026-09-19-web-smoke.md` and the WebGPU wiring
status in `crates/t0-wasm/README.md`.
