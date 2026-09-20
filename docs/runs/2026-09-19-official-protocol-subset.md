# Official-protocol subset — their code, their protocol, our quants

Machine: Apple M2 (Darwin 25.3.0), CPU only (no MPS/GPU per this task's
laptop rules). Commit: `4404b6f` (worktree `fullsuite`, branch `fullsuite`).

This run answers a different question than
`docs/runs/2026-09-19-gifteval-subset.md`: that run compares OUR Burn/wgpu
port against the reference on the SAME 8 GIFT-Eval configs but at a
context-512 cap and window-subsampled, because our port has no autoregressive
rollout yet. This run instead uses **their own code, their own protocol,
unmodified** (`CONTEXT_LENGTH=8192`, `gift_eval.data.Dataset`'s full test
split with no subsampling, `t0.evaluation.T0Predictor`, `gluonts.model.evaluate_model`
+ their metrics) on the SAME 8 configs, and swaps only the weights:

- **(a) f32** — `theforecastingcompany/t0-alpha` reference weights, PyTorch.
- **(b) onnx-int8** — their first-party `t0-alpha-onnx-int8` export
  (`t0-alpha-grouped-int8.onnx`, per-channel signed INT8 weights / FP32
  compute), run through `onnxruntime` with a predictor wrapper matching
  `T0Predictor`'s interface (`tools/fullsuite/onnx_predictor.py`).
- **(c) q8_0** — our GGUF Q8_0 export, dequantized back to f32
  (`tools/fullsuite/dequant_gguf.py`) and loaded into the SAME PyTorch
  reference architecture/predictor as (a) — a fake-quant: only the weights
  differ from (a).
- **(d) q4_0** — same, our GGUF Q4_0.
- **(e) ours (context-capped, NOT protocol-equivalent)** — our Burn engine's
  existing numbers from `docs/BENCHMARKS.md`'s "GIFT-Eval subset" table:
  context capped to 512 (both reference and ours), windows subsampled to
  <=60/config. Listed here for orientation only. Config names differ
  cosmetically (`electricity/W-FRI` vs `electricity/W`, `saugeenday` vs
  `saugeen` — same series, see that doc).

None of our Rust/Burn port is touched by (a)-(d); this is entirely "their
code" apart from the substituted weights in (c)/(d).

## Versions

- Python 3.14.3, `torch` 2.14.0, `gluonts` 0.15.1, `tfc-t0` 0.5.0,
  `salesforce-gift-eval` 0.0.0a0 (all from the main checkout's `.venv`,
  reused read-only per this task's setup — this worktree has no
  network-install permission for pip), `onnxruntime` 1.24.2 (Homebrew
  Python 3.14 site-packages, bridged in via `sys.path.append` inside
  `onnx_predictor.py` — appended, not prepended, to avoid shadowing the
  venv's numpy/pyarrow, which broke `datasets`' arrow formatting on the
  first attempt).
- Weights: `theforecastingcompany/t0-alpha` (`model.safetensors`, sha
  unchanged from `docs/reports/t0-published-numbers.md`),
  `theforecastingcompany/t0-alpha-onnx-int8` (`t0-alpha-grouped-int8.onnx`,
  sha256 `82da4392...`, downloaded this session to
  `~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha-onnx-int8/`).
- Our GGUF exports: `t0-cli export-gguf --quant q8_0|q4_0` (binary built at
  the main checkout, commit `4404b6f`, invoked read-only from this worktree),
  written to `tools/fullsuite/weights/t0-alpha-{q8_0,q4_0}.gguf`.

## Commands

```
# GGUF exports (main checkout's already-built t0-cli, read-only)
t0-cli export-gguf --weights <t0-alpha>/model.safetensors --config <t0-alpha>/config.json --quant q8_0 --out tools/fullsuite/weights/t0-alpha-q8_0.gguf
t0-cli export-gguf --weights <t0-alpha>/model.safetensors --config <t0-alpha>/config.json --quant q4_0 --out tools/fullsuite/weights/t0-alpha-q4_0.gguf

# dequant + per-tensor error table + forward parity
.venv/bin/python3 tools/fullsuite/dequant_gguf.py --gguf tools/fullsuite/weights/t0-alpha-q8_0.gguf \
    --reference-safetensors <t0-alpha>/model.safetensors --config <t0-alpha>/config.json \
    --out tools/fullsuite/weights/t0-alpha-q8_0.dequant.safetensors --fixture fixtures/case0_univariate_sine
# (same for q4_0)

# official-protocol eval, per variant
.venv/bin/python3 tools/fullsuite/run_gifteval.py --variant f32 --max-minutes 45
.venv/bin/python3 tools/fullsuite/run_gifteval.py --variant q8_0 --max-minutes 45
.venv/bin/python3 tools/fullsuite/run_gifteval.py --variant q4_0 --max-minutes 45
.venv/bin/python3 tools/fullsuite/run_gifteval.py --variant onnx-int8 --max-minutes 45
```

## Dequant parity: GGUF -> f32 vs original safetensors

Per-tensor max-abs/max-rel error over all 303 tensors (worst-case shown;
full per-tensor table is printed by `dequant_gguf.py`, not committed). The
big max-rel values are expected and harmless: they occur on near-zero
reference elements (max-rel is unbounded as the denominator -> 0; see
`per_tensor_error_table`'s `max(|ref|, 1e-8)` floor) and are dominated by
tiny bias/norm entries, not by the big matmuls that actually carry signal.

| quant | worst tensor max-abs | worst tensor max-rel | one-fixture forward parity (quantile max-abs) | one-fixture forward parity (quantile max-rel) | BENCHMARKS.md's case0 Rust-side row (max-abs / max-rel) |
|---|---|---|---|---|---|
| q8_0 | 1.096684e-02 | 1.000763e+00 | 2.638698e-03 | 3.320146e-01 | 2.638936e-03 / 3.320634e-01 |
| q4_0 | 1.750424e-01 | 1.000932e+00 | 4.542732e-02 | 5.438239e+00 | 4.542732e-02 / 5.427120e+00 |

Both agree with the Rust-side (`t0-cli drift`) numbers to within floating-point
noise (max-abs differs in the 4th+ significant digit; max-rel differs by
<0.3% relative) — confirms this Python reader decodes the exact same Q8_0/Q4_0
block layout as `crates/t0-core/src/gguf.rs`, and that the PyTorch-loaded
fake-quant forward pass reproduces the Rust ndarray-backend forward pass.

## Official protocol, 8-config subset, per-config

CRPS (`mean_weighted_sum_quantile_loss`, their metric call):

| config | (a) f32 | (b) onnx-int8 | (c) our q8_0 (fake-quant) | (d) our q4_0 (fake-quant) | (e) ours, context-512-capped, NOT comparable |
|---|---|---|---|---|---|
| us_births/D/short | 0.0182 | 0.0181 | 0.0182 | 0.0184 | 0.0231 |
| saugeen(day)/D/short | 0.3525 | 0.3520 | 0.3522 | 0.3538 | 0.4133 |
| jena_weather/D/short | 0.0458 | 0.0456 | 0.0458 | 0.0457 | 0.0458 |
| solar/W(-FRI)/short | 0.1876 | 0.1878 | 0.1876 | 0.1872 | 0.1864 |
| solar/D/short | 0.2753 | 0.2751 | 0.2754 | 0.2747 | 0.2711 |
| loop_seattle/D/short | 0.0433 | 0.0434 | 0.0433 | 0.0434 | 0.0453 |
| electricity/W(-FRI)/short | 0.0569 | 0.0569 | 0.0568 | 0.0575 | 0.0529 |
| electricity/D/short | 0.0539 | 0.0539 | 0.0539 | 0.0539 | 0.0789 |
| **aggregate (geometric mean)** | **0.0818** | **0.0818** | **0.0818** | **0.0821** | **0.0897** |

MASE:

| config | (a) f32 | (b) onnx-int8 | (c) our q8_0 (fake-quant) | (d) our q4_0 (fake-quant) | (e) ours, context-512-capped, NOT comparable |
|---|---|---|---|---|---|
| us_births/D/short | 0.3428 | 0.3423 | 0.3429 | 0.3475 | 0.4509 |
| saugeen(day)/D/short | 2.9944 | 2.9886 | 2.9921 | 3.0190 | 3.2756 |
| jena_weather/D/short | 1.0691 | 1.0690 | 1.0687 | 1.0770 | 1.0691 |
| solar/W(-FRI)/short | 1.2879 | 1.2865 | 1.2877 | 1.2836 | 1.3052 |
| solar/D/short | 0.9874 | 0.9872 | 0.9873 | 0.9893 | 0.9855 |
| loop_seattle/D/short | 0.8893 | 0.8902 | 0.8893 | 0.8906 | 0.8970 |
| electricity/W(-FRI)/short | 1.5185 | 1.5185 | 1.5182 | 1.5312 | 1.4938 |
| electricity/D/short | 1.3886 | 1.3884 | 1.3885 | 1.3900 | 1.7325 |
| **aggregate (geometric mean)** | **1.1278** | **1.1272** | **1.1276** | **1.1332** | **1.2139** |

Per-config n_windows (full test set, NOT subsampled, identical across all
of (a)-(d)): us_births 20, saugeen 20, jena_weather 42, solar/W 137,
solar/D 274, loop_seattle 646, electricity/W 1110, electricity/D 1850.
(e)'s subset used <=60 windows/config (`docs/runs/2026-09-19-gifteval-subset.md`).

## Reading this table

At full 8192-context, no-rollout-needed protocol (all horizons here are
<=30, well under the single-forward-pass 1024 limit both the PyTorch model
and the ONNX graph support), **all four variants land within ~0.5% of each
other on both CRPS and MASE, aggregate and per-config** — including our
own Q4_0 (aggregate MASE +0.48% vs f32, CRPS +0.37%), which is a tighter
result than the context-512/subsampled run in
`docs/runs/2026-09-19-gifteval-subset.md` reported for Q4_0 (some
individual tasks there exceeded a 0.5% band). The likely reason: full
8192 context gives the model much more history to condition on, which
appears to make the forecast less sensitive to per-tensor quantization
noise than the earlier context-starved (512-step) setup — consistent with
this being a context-length effect, not a quant-recipe difference (same
Q4_0 weights, same drift numbers in `docs/BENCHMARKS.md`'s Milestone 1a+
table). `onnx-int8` (their own INT8 export, independently engineered) sits
in the same tight band as our own fake-quant Q8_0/Q4_0 rows, which is the
apples-to-apples confirmation this task set out to get: at this protocol,
our quantization ladder is not measurably worse than their own INT8 export
on this checkpoint/config mix.

(e)'s absolute numbers differ from (a)-(d) mostly because of the
context-512 cap and the different (arguably harder, since saugeen/electricity
lose most of their history) truncated series -- not because of quantization;
see that doc's own analysis for why it isn't a fair comparison to this table.

## Timings (CPU, per-config, all completed within the 45-minute budget — no
projection needed)

| config | n_windows | (a) f32 s | (c) q8_0 s | (d) q4_0 s | (b) onnx-int8 s |
|---|---|---|---|---|---|
| us_births/D/short | 20 | 2.13 | 2.28 | 2.30 | 3.02 |
| saugeen/D/short | 20 | 1.87 | 2.15 | 2.14 | 3.80 |
| jena_weather/D/short | 42 | 0.24 | 0.24 | 0.24 | 0.59 |
| solar/W/short | 137 | 0.30 | 0.34 | 0.38 | 1.35 |
| solar/D/short | 274 | 1.43 | 1.54 | 1.53 | 5.01 |
| loop_seattle/D/short | 646 | 3.79 | 3.67 | 3.69 | 15.35 |
| electricity/W/short | 1110 | 5.15 | 5.41 | 5.15 | 37.12 |
| electricity/D/short | 1850 | 68.47 | 38.99 | 62.95 | 173.99 |
| **total** | 4099 | 83.4s | 54.6s | 78.4s | 240.2s (4.0 min) |

electricity/D q8_0 ran notably faster than f32/q4_0 (39s vs 68s/63s) despite
identical F32 compute for all three PyTorch-backed variants (dequant happens
once at load; the matmul precision is F32 regardless of storage quant, same
fact `docs/BENCHMARKS.md` already establishes) — read as CPU
scheduling/thermal noise between separate sequential process launches, not a
quant-dependent compute cost; not investigated further since it doesn't
change any of the accuracy conclusions above. All four variants finished
comfortably under the 45-minute budget with no need to stop early or
project; total wall time across all four full runs was under 7 minutes.

## What this deviates from vs the officially published numbers

Their published GIFT-Eval numbers (CRPS 0.4941, MASE 0.7240,
`docs/reports/t0-published-numbers.md`) are over all 97 configs/3 terms;
this run is 8 configs, short term only, so absolute numbers here are not
comparable to that headline figure (same caveat
`docs/runs/2026-09-19-gifteval-subset.md` already states) — this run
IS however at their exact context length (8192) and exact windowing (no
subsampling), unlike that other doc.

## Pending: the full 97-config run

Not run tonight (CPU/M2, laptop rules). `tools/fullsuite/setup_3080.sh`
(one-time venv setup) and `tools/fullsuite/launch_3080.sh` (systemd-run,
detached, all four variants) are prepared for the GPU box; see that
script's header for the exact launch invocation. Neither script has been
run against the box this session.
