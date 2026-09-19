# Single-signal native CPU latency, per quant

- Machine: Apple M2 (Darwin 25.3.0), 8 threads available (`std::thread::available_parallelism`).
- Commit: `01257dc` (`cli: add --warmup flag to bench`).
- GPU: occupied by `llm-life train-a` (a fine-tune in a sibling repo,
  `ps aux` confirmed running throughout this run) — per this run's hard
  constraint, no wgpu/Metal/WebGPU measurement was taken; ndarray (CPU)
  backend only. Note: `llm-life train-a` is itself a CPU/GPU-mixed process
  (data loading, optimizer step) sharing the same 8 logical cores as this
  CPU-only bench, so these numbers carry more run-to-run noise than an
  idle-machine measurement would; the per-quant *ranking* is still the
  signal of interest, not the absolute ms.
- Command (per row): `t0-cli bench --weights <F32 safetensors | GGUF> [--config config.json] --backend ndarray --signals 1 --context 512 --horizon 32 --warmup 2 --reps 10` — median of 10 timed forward passes after 2 untimed warm-ups.
- Weights: `../models/hf/theforecastingcompany/t0-alpha/{model.safetensors,config.json}` (F32); `/tmp/t0-alpha-{f16,q8_0,q4_0}.gguf` (already exported this session via `t0-cli export-gguf`, config embedded in the GGUF).

| quant | file MB | load time (s) | median forward latency (ms), 1 signal, ctx 512, horizon 32 |
|---|---|---|---|
| F32 | 406.6 | 0.669 | 363.2 |
| f16 (load-time cast) | 203.3 | 0.799 | 419.9 |
| Q8_0 | 108.9 | 0.543 | 398.0 |
| Q4_0 | 58.6 | 0.362 | 424.2 |

Raw reps (all 10, seconds) per row:

- F32: `[0.4228, 0.3530, 0.3543, 0.3587, 0.3632, 0.3664, 0.3511, 0.3569, 0.3722, 0.4038]`
- f16: `[0.5073, 0.4721, 0.4526, 0.4238, 0.4199, 0.4005, 0.3910, 0.4092, 0.3759, 0.4072]`
- Q8_0: `[0.5222, 0.5696, 0.5903, 0.4988, 0.3980, 0.3582, 0.3624, 0.3768, 0.3597, 0.3611]`
- Q4_0: `[0.3717, 0.4342, 0.3698, 0.4047, 0.4462, 0.4390, 0.4242, 0.4171, 0.4190, 0.4248]`

Observation: load time scales with file size as expected (F32 > f16 >
Q8_0 > Q4_0). Forward latency does **not** improve with smaller quant at
n=1 signal, consistent with the wgpu chunked-batch finding already logged
in `docs/BENCHMARKS.md` (compute stays F32 regardless of storage quant —
dequant happens once at load, not per forward pass) and with the
CPU-timing-noise finding in `docs/runs/2026-09-19-milestone1a.md`; here
that noise is compounded by `llm-life`'s concurrent CPU use. F32 reads as
fastest median only because its first few reps happened to land in a
quieter window — not read as a genuine quant-dependent speedup.

## Browser / WebGPU

Pending — GPU busy with `llm-life train-a` per this run's hard constraint.
Marked "pending (GPU busy)" in `docs/BENCHMARKS.md`.
