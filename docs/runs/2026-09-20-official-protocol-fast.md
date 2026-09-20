# Official-protocol subset — our `t0-fast` engine, same 8 configs

Machine: Apple M2 (Darwin 25.3.0), native Metal (`wgpu`), CPU data
loading + GPU forward via `crates/t0-fast`. Commits: `4979e48` (rollout
underflow fix), `fdca2a8` (`cmd_gifteval_fast` safetensors+`fast_quant`
support), `f2cf870` (bind-group cache key fix) — see
`docs/runs/2026-09-20-rollout.md`'s follow-up section for both bugs.

This is the "ours, same protocol" half that
`docs/runs/2026-09-19-official-protocol-subset.md` (their PyTorch/ONNX
code, our GGUF weights fake-quanted back into their architecture) never
touches our Rust port for: same 8 GIFT-Eval configs, same protocol
(`CONTEXT_LENGTH≈8192`, full test split, no window subsampling), but
forecast through `t0-cli gifteval --backend fast` (`crates/t0-fast`, raw
`wgpu`/WGSL, no Burn at inference) instead of their PyTorch model. This
was blocked until this session because the official protocol's longest
windows (`saugeenday`, `us_births`) need more than one forward pass' worth
of patches at `max_horizon`-sized single windows in some configs and,
before the rollout-underflow fix above, `forecast_rollout` never
terminated on any backend. In fact none of these 8 configs' horizons
(<=30) actually exceed `max_horizon=1024`, so no AR rollout is needed here
after all — every window here is a single `t0_fast::forecast` call; the
rollout fix mattered because it was blocking trust in the rest of this
session's fast-backend work, not because this particular subset needed it.

## Deviation: context 8160, not 8192

`t0-fast`'s attention kernel (`crates/t0-fast/src/shaders/attention.wgsl`)
caps the patch count attended jointly by one time-layer workgroup at
`MAX_SEQ=256` (WebGPU's workgroup-invocation limit). One forward pass
attends `ceil(context/32) + ceil(horizon/32)` patches; `CONTEXT_LENGTH=8192`
is exactly 256 patches, which plus even a 1-patch horizon exceeds the cap.
`tools/gifteval_official_export.py` uses **8160** (255 patches) instead.
Of the 8 configs, only `saugeenday/D` has full history longer than 8192
samples (up to 23711) and is actually affected — by 32 fewer of its
*oldest* samples, out of >20000. The other 7 configs' full available
history (305–7275 samples, see table below) is already well under 8160, so
their windows are identical to the true 8192-context protocol regardless
(the cap never binds).

Per-config max available context (from `tools/gifteval_official_export.py`'s
run): us_births 7275, saugeenday 8160 (capped, true max 23711), jena_weather
336, solar/W 44, solar/D 335, loop_seattle 335, electricity/W 200,
electricity/D 1431.

## Commands

```
.venv/bin/python3 tools/gifteval_official_export.py --out-dir /tmp/gifteval_official

t0-cli export-gguf --weights <t0-alpha>/model.safetensors --config <t0-alpha>/config.json --quant q8_0 --out /tmp/t0_gguf/t0-alpha-q8_0.gguf
t0-cli export-gguf --weights <t0-alpha>/model.safetensors --config <t0-alpha>/config.json --quant q4_0 --out /tmp/t0_gguf/t0-alpha-q4_0.gguf

t0-cli gifteval --backend fast --fast-quant f32 --manifest /tmp/gifteval_official \
    --weights <t0-alpha>/model.safetensors --config <t0-alpha>/config.json \
    --out-dir /tmp/gifteval_official/forecasts_f32
t0-cli gifteval --backend fast --weights /tmp/t0_gguf/t0-alpha-q8_0.gguf \
    --manifest /tmp/gifteval_official --out-dir /tmp/gifteval_official/forecasts_q8_0
t0-cli gifteval --backend fast --weights /tmp/t0_gguf/t0-alpha-q4_0.gguf \
    --manifest /tmp/gifteval_official --out-dir /tmp/gifteval_official/forecasts_q4_0

.venv/bin/python3 tools/score_official_protocol.py --variant f32 \
    --manifest-dir /tmp/gifteval_official --forecast-dir /tmp/gifteval_official/forecasts_f32
# (same for q8_0, q4_0)
```

Manifest + per-window `.f32` context/future files and forecast outputs are
regenerable data (like `fixtures/gifteval/`), written to `/tmp` this
session rather than committed.

## Results: our `t0-fast` engine vs their code, per config

Per-config n_windows (full test set, no subsampling, identical across all
three of our variants and all four of theirs): us_births 20, saugeenday
20, jena_weather 42, solar/W 137, solar/D 274, loop_seattle 646,
electricity/W 1110, electricity/D 1850 (4099 total). "theirs" columns are
`docs/runs/2026-09-19-official-protocol-subset.md`'s (a) f32 and (c)/(d)
our-GGUF-weights-fake-quanted-into-their-PyTorch-architecture rows — the
same weights as our (f32/Q8_0/Q4_0) columns here, different engine.

CRPS (`mean_weighted_sum_quantile_loss`):

| config | our engine f32 | their code f32 | our engine Q8_0 | their code Q8_0 (fake-quant) | our engine Q4_0 | their code Q4_0 (fake-quant) |
|---|---|---|---|---|---|---|
| us_births/D/short | 0.0182 | 0.0182 | 0.0182 | 0.0182 | 0.0184 | 0.0184 |
| saugeenday/D/short | 0.3519 | 0.3525 | 0.3517 | 0.3522 | 0.3528 | 0.3538 |
| jena_weather/D/short | 0.0458 | 0.0458 | 0.0458 | 0.0458 | 0.0457 | 0.0457 |
| solar/W(-FRI)/short | 0.1876 | 0.1876 | 0.1876 | 0.1876 | 0.1872 | 0.1872 |
| solar/D/short | 0.2753 | 0.2753 | 0.2754 | 0.2754 | 0.2747 | 0.2747 |
| loop_seattle/D/short | 0.0433 | 0.0433 | 0.0433 | 0.0433 | 0.0434 | 0.0434 |
| electricity/W(-FRI)/short | 0.0569 | 0.0569 | 0.0568 | 0.0568 | 0.0575 | 0.0575 |
| electricity/D/short | 0.0539 | 0.0539 | 0.0539 | 0.0539 | 0.0539 | 0.0539 |
| **aggregate (geometric mean)** | **0.0818** | **0.0818** | **0.0818** | **0.0818** | **0.0820** | **0.0821** |

MASE:

| config | our engine f32 | their code f32 | our engine Q8_0 | their code Q8_0 (fake-quant) | our engine Q4_0 | their code Q4_0 (fake-quant) |
|---|---|---|---|---|---|---|
| us_births/D/short | 0.3428 | 0.3428 | 0.3429 | 0.3429 | 0.3475 | 0.3475 |
| saugeenday/D/short | 2.9921 | 2.9944 | 2.9896 | 2.9921 | 3.0124 | 3.0190 |
| jena_weather/D/short | 1.0691 | 1.0691 | 1.0686 | 1.0687 | 1.0770 | 1.0770 |
| solar/W(-FRI)/short | 1.2879 | 1.2879 | 1.2877 | 1.2877 | 1.2836 | 1.2836 |
| solar/D/short | 0.9874 | 0.9874 | 0.9873 | 0.9873 | 0.9893 | 0.9893 |
| loop_seattle/D/short | 0.8893 | 0.8893 | 0.8893 | 0.8893 | 0.8906 | 0.8906 |
| electricity/W(-FRI)/short | 1.5185 | 1.5185 | 1.5182 | 1.5182 | 1.5312 | 1.5312 |
| electricity/D/short | 1.3886 | 1.3886 | 1.3885 | 1.3885 | 1.3900 | 1.3900 |
| **aggregate (geometric mean)** | **1.1277** | **1.1278** | **1.1274** | **1.1276** | **1.1329** | **1.1332** |

**Our engine matches their code to the 3rd-4th significant digit on every
config, both metrics, all three weight residencies** — the small
per-config differences (e.g. `saugeenday` CRPS 0.3519 vs 0.3525 at f32)
are attributable to the 8160-vs-8192 context deviation on that one config
(more history for the reference run, everything else identical), not to
any divergence in the forward pass itself (the standard + long-rollout
parity fixtures above already establish <2e-6 max-abs agreement against
Burn's own reference at matched inputs).

## Timings (per config, our `t0-fast` engine, M2 Metal, single window at a time)

| config | n_windows | F32 s (ms/window) | Q8_0 s (ms/window) | Q4_0 s (ms/window) |
|---|---|---|---|---|
| us_births/D/short | 20 | 8.88 (444.0) | 8.72 (435.9) | 8.97 (448.6) |
| saugeenday/D/short | 20 | 9.78 (488.8) | 9.70 (485.2) | 10.06 (502.9) |
| jena_weather/D/short | 42 | 2.49 (59.3) | 3.12 (74.2) | 2.37 (56.3) |
| solar/W/short | 137 | 6.30 (46.0) | 6.93 (50.6) | 5.27 (38.5) |
| solar/D/short | 274 | 16.19 (59.1) | 20.43 (74.5) | 17.46 (63.7) |
| loop_seattle/D/short | 646 | 38.11 (59.0) | 66.34 (102.7) | 41.39 (64.1) |
| electricity/W/short | 1110 | 55.28 (49.8) | 72.92 (65.7) | 55.99 (50.4) |
| electricity/D/short | 1850 | 247.85 (134.0) | 285.78 (154.5) | 255.17 (137.9) |
| **total** | 4099 | **384.9s (6.4 min)** | **473.9s (7.9 min)** | **396.7s (6.6 min)** |

us_births/saugeenday are far slower per-window (~450-500ms) than the rest
(~40-155ms) because they carry the largest context (7275/8160 samples,
~228/255 patches) — one forward pass there attends over that many patches
jointly in every time layer, vs <=45 patches for the other 6 configs. Note
these single-window numbers are not directly comparable to
`docs/BENCHMARKS.md`'s "warm ms" latency row (that's a 512-context,
32-horizon window with the model already resident; here weight-load is a
one-time 0.05-0.18s at the top of each run, not per-window). Q8_0 running
slower than F32 here (unlike the batch-24 bench numbers in
`docs/BENCHMARKS.md`, where quantized rows beat F32) is consistent with
that doc's open note that single-call (`M=1`-scale) rows stay on the naive
kernel where dequant cost isn't amortized — most windows here are
single-signal too.

## Projected 97-config wall time (not run)

This subset is 8 of the ~97 GIFT-Eval configs, short term only. Assuming
the other 89 short-term configs have a similar per-window-cost and
window-count distribution to this subset (an assumption, not a
measurement — some configs are known to run at coarser frequencies or
different history lengths, e.g. sub-hourly configs excluded from this
subset for download-size reasons per `docs/runs/2026-09-19-gifteval-subset.md`,
which would shift both context length and window count): linearly scaling
this subset's total time by `97/8 = 12.125x` gives:

| variant | this subset (8 configs) | projected (97 configs, linear scale) |
|---|---|---|
| F32 | 384.9s (6.4 min) | ~4670s (~78 min) |
| Q8_0 | 473.9s (7.9 min) | ~5750s (~96 min) |
| Q4_0 | 396.7s (6.6 min) | ~4810s (~80 min) |
| all three | 1255.5s (20.9 min) | ~15230s (~4.2 h) |

This is an order-of-magnitude estimate for planning purposes, not a
commitment — the full 97-config run was explicitly out of scope for this
session and was not attempted.
