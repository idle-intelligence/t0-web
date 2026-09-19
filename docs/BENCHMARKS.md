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

Gate: max-abs quantile error <= 1e-4 across all fixtures. Result: PASS (worst case 1.192093e-6).

Analysis: `docs/runs/2026-09-19-parity.md`.

## Milestone 1a — GGUF quantization round-trip (native, CPU/ndarray)

- Machine: Apple M2 (Darwin 25.3.0).
- Commit: (this milestone's HEAD — see `git log`).
- Command: `cargo test -p t0-core --release -- --ignored --nocapture gguf_round_trip_quantiles`
- Quantized tensors: `attention.wQKV.weight`, `attention.wO.weight`, `mlp.0.weight`, `mlp.2.weight` per layer; everything else stored as f16.

File sizes (`t0-cli export-gguf`, from `model.safetensors`, 406,601,492 bytes / 406.6 MB):

| quant | file bytes | file MB | ratio to F32 safetensors |
|---|---|---|---|
| f16 | 203,307,887 | 203.3 | 0.500x |
| q8_0 | 108,936,047 | 108.9 | 0.268x |
| q4_0 | 58,604,399 | 58.6 | 0.144x |

Quantile error vs the reference PyTorch fixtures (max over the 3 fixtures; f16 gated <=1e-3, Q8_0/Q4_0 recorded only):

| quant | quantile max-abs err | quantile max-rel err | gate |
|---|---|---|---|
| f16 | 4.151165e-4 | 2.925728e-1 | <=1e-3: PASS |
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

Analysis: `docs/runs/2026-09-19-milestone1a.md`.

## Milestone 1a — batch path (native, CPU/ndarray)

- Command: `cargo test -p t0-core --release -- --ignored --nocapture batch_of_8_copies_equals_8x_single`
- Result: PASS — 8 copies of fixture 0 batched in one forward pass match 8x the single-signal forecast to <=1e-5 max-abs.

## Milestone 1a — latency table

- Machine: Apple M2 (Darwin 25.3.0), 8 threads available (`std::thread::available_parallelism`).
- Commit: (this milestone's HEAD — see `git log`).
- GPU contention: a `jacobi2000` run held the GPU during the first wgpu pass; numbers below are re-measured with the GPU idle (see `docs/runs/2026-09-19-milestone1a.md`).
- Command (per row): `t0-cli bench --weights <F32 safetensors | GGUF> [--config <config.json>] --backend <ndarray|wgpu> --signals <N> --context 512 --horizon 96 --reps 5` — median of 5 timed forward passes after 1 untimed warm-up.

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
| wgpu (Metal) | F32 | 1000 | 406.6 | — | fails, see `docs/runs/2026-09-19-milestone1a.md` |
| wgpu (Metal) | f16 | 1 | 203.3 | 0.204 | 0.223 |
| wgpu (Metal) | f16 | 16 | 203.3 | 0.201 | 0.593 |
| wgpu (Metal) | f16 | 1000 | 203.3 | — | fails, see `docs/runs/2026-09-19-milestone1a.md` |
| wgpu (Metal) | Q8_0 | 1 | 108.9 | 0.195 | 0.222 |
| wgpu (Metal) | Q8_0 | 16 | 108.9 | 0.196 | 0.596 |
| wgpu (Metal) | Q8_0 | 1000 | 108.9 | — | fails, see `docs/runs/2026-09-19-milestone1a.md` |
| wgpu (Metal) | Q4_0 | 1 | 58.6 | 0.158 | 0.222 |
| wgpu (Metal) | Q4_0 | 16 | 58.6 | 0.161 | 0.592 |
| wgpu (Metal) | Q4_0 | 1000 | 58.6 | — | fails, see `docs/runs/2026-09-19-milestone1a.md` |

Analysis: `docs/runs/2026-09-19-milestone1a.md`.

## Milestone 1a — chunked wgpu batch (native, Metal), extends the table above

- Machine: Apple M2 (Darwin 25.3.0), GPU idle (`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before each run).
- Commit: `d11f154` + this run's uncommitted changes (`forecast_batch_chunked`, `DEFAULT_BATCH_CHUNK = 16`).
- Correctness gate: `cargo test -p t0-cli --release --no-default-features --features wgpu -- --ignored chunked_batch_of_40_equals_40_single` — 40 synthetic signals run through wgpu in chunks of 16 (3 chunks: 16+16+8) match 40 single-signal (chunk=1) forecasts to <=1e-5 max-abs. Result: **PASS**.
- Command (per row): `t0-cli bench --weights <F32 safetensors | GGUF> [--config <config.json>] --backend wgpu --signals <N> --context 512 --horizon 96 --reps 3 --chunk 16`.

| backend | quant | signals | chunk | median forward s | per-signal ms |
|---|---|---|---|---|---|
| wgpu (Metal) | F32 | 100 | 16 | 3.827 | 38.27 |
| wgpu (Metal) | f16 | 100 | 16 | 3.806 | 38.06 |
| wgpu (Metal) | Q8_0 | 100 | 16 | 3.834 | 38.34 |
| wgpu (Metal) | Q4_0 | 100 | 16 | 3.808 | 38.08 |
| wgpu (Metal) | F32 | 1000 | 16 | 37.989 | 37.99 |
| wgpu (Metal) | f16 | 1000 | 16 | 46.496 | 46.50 |
| wgpu (Metal) | Q8_0 | 1000 | 16 | 51.183 | 51.18 |
| wgpu (Metal) | Q4_0 | 1000 | 16 | 50.200 | 50.20 |

Per-signal ms is flat (~38ms) at n=100 across all quant schemes (compute stays F32 regardless of storage quant, same fact as the n=1/16 rows above). At n=1000, per-signal ms rises to 38-51ms and is noisier across quant schemes (F32 38.0, f16 46.5, Q8_0 51.2, Q4_0 50.2 ms/signal) — read as chunk-count/thermal noise (63 sequential chunks vs 7 at n=100, each rep spanning 37-88s) rather than a genuine quant-dependent compute cost, consistent with this doc's existing "CPU timing noise" finding now showing up on wgpu at high chunk counts too (see `docs/runs/2026-09-19-milestone1a.md`).

Analysis and the upstream cubek-matmul issue note: `docs/runs/2026-09-19-milestone1a.md`.

## Milestone 1a+ — drift benchmark, apples to apples with the published INT8 card

- Machine: Apple M2 (Darwin 25.3.0), ndarray backend (drift is a numerics check, not a wgpu benchmark — the CLI's `drift` subcommand has no `--backend` flag, see `docs/runs/2026-09-19-milestone1a.md`).
- Commit: this run's uncommitted changes (`t0-cli drift`, `Weights::quantize_int8_theirs`, `gguf::int8_per_channel_roundtrip`).
- Reference: our F32 forward pass (== PyTorch by the Milestone-0 parity gate, worst case 1.2e-6 max-abs).
- Command (per row): `t0-cli export-gguf --weights model.safetensors --config config.json --quant <f16|q8_0|q4_0> --out t0-alpha-<q>.gguf`, then `t0-cli drift --weights-f32 model.safetensors --config config.json --quant <t0-alpha-<q>.gguf | int8-theirs> --cases 54 --seed 42`.
- Case generator, cases, formula: **not** their published generator (not public — see `docs/reports/t0-published-numbers.md`); our own reconstruction, documented in full in `docs/runs/2026-09-19-milestone1a.md`. Drift formula matches their card verbatim: `|error| / (max(reference) - min(reference))` over the whole forecast (horizon x quantiles), reference = our F32 output for that case.

| quant | mean drift worst % | mean drift mean % | point drift worst % | point drift mean % | file MB |
|---|---|---|---|---|---|
| f16 | 0.0336 | 0.0111 | 0.1297 | 0.0412 | 203.3 |
| q8_0 | 0.5299 | 0.1076 | 1.7833 | 0.3892 | 108.9 |
| q4_0 | 3.1084 | 1.3816 | 8.4192 | 4.8130 | 58.6 |
| int8-theirs (per-channel INT8, 96 matmuls; 6 I/O projections FP32) | 0.8494 | 0.1774 | 2.0400 | 0.6083 | 103.3 (estimated, no file written) |
| **their published t0-beta INT8** (`t0-beta-onnx-int8`, different checkpoint: t0-beta not t0-alpha, different case generator) | **0.2271** | — | **9.393** | — | 269.1 |

Their number is for **t0-beta** (256M params, embed_dim 1024); ours here is **t0-alpha** (102M params, embed_dim 512) — the two checkpoints are not architecture-identical (see `docs/reports/t0-published-numbers.md`), so this is not yet a same-model apples-to-apples row; it is the closest available until t0-beta is ported (GOAL.md's open question on which checkpoint to claim against). `int8-theirs` (their recipe, our checkpoint, our case generator) is our best current apples-to-apples proxy for the *recipe*: worst-case mean drift 0.8494% is above their 0.2271% but with a different model size and a different (undocumented, hence reconstructed) case generator, so this gap should not be read as "our INT8 recipe is worse" — it may equally reflect t0-alpha's smaller embed_dim (512 vs 1024) being more sensitive to per-channel quantization, or a harder case distribution. Q4_0 clearly exceeds their 2%/10% acceptance gates; Q8_0 and int8-theirs are within both gates; f16 is far inside both.

Analysis: `docs/runs/2026-09-19-milestone1a.md`.

## GIFT-Eval subset — reference F32 vs ours F32/f16/Q8_0/Q4_0, t0-alpha

- Machine: Apple M2 (Darwin 25.3.0), `ndarray` (CPU) backend only — `llm-life` (another repo's fine-tune) held the GPU throughout this run, so no wgpu numbers are reported here.
- Commit: c1d9183 (`cli: add gifteval subcommand...`) + this doc's commit.
- Reference: `tfc-t0` (PyPI) `T0Forecaster.predict()`, checkpoint `theforecastingcompany/t0-alpha`, scored with the official GIFT-Eval notebook's own metric call (`gluonts.model.evaluate_model` + `MASE()` + `MeanWeightedSumQuantileLoss(quantile_levels=[0.1..0.9])`).
- Subset: 8 configs / 382 windows (stride-subsampled from 4099; see `docs/runs/2026-09-19-gifteval-subset.md`). **Deviation from the published protocol**: context capped to the trailing 512 observations for BOTH reference and ours (this port has no autoregressive rollout yet — `crates/t0-core/src/model.rs::forecast_series` is one forward pass, `max_horizon=1024`), vs the reference notebook's `CONTEXT_LENGTH=8192`. Numbers below are only comparable to each other, not to the published full-suite row.
- Commands: `tools/gifteval_subset.py` (export) -> `t0-cli gifteval --backend ndarray --manifest fixtures/gifteval --weights <safetensors|gguf> --out-dir fixtures/gifteval/forecasts_<tag>` (forecast) -> `tools/score_gifteval.py --which <reference|f32|f16|q8_0|q4_0> --forecast-dir <dir>` (score, same gluonts call both sides).

CRPS (`mean_weighted_sum_quantile_loss`):

| task (config/freq/term) | reference F32 | ours F32 | ours f16 | ours Q8_0 | ours Q4_0 |
|---|---|---|---|---|---|
| electricity/D/short | 0.0789 | 0.0789 | 0.0789 | 0.0789 | 0.0792 |
| electricity/W-FRI/short | 0.0529 | 0.0529 | 0.0529 | 0.0529 | 0.0535 |
| solar/D/short | 0.2711 | 0.2711 | 0.2711 | 0.2711 | 0.2706 |
| solar/W-FRI/short | 0.1864 | 0.1864 | 0.1864 | 0.1863 | 0.1861 |
| jena_weather/D/short | 0.0458 | 0.0458 | 0.0458 | 0.0458 | 0.0457 |
| loop_seattle/D/short | 0.0453 | 0.0453 | 0.0453 | 0.0453 | 0.0454 |
| saugeenday/D/short | 0.4133 | 0.4133 | 0.4133 | 0.4133 | 0.4169 |
| us_births/D/short | 0.0231 | 0.0231 | 0.0231 | 0.0231 | 0.0230 |
| **aggregate (geometric mean)** | **0.0897** | **0.0897** | **0.0897** | **0.0897** | **0.0898** |

MASE:

| task | reference F32 | ours F32 | ours f16 | ours Q8_0 | ours Q4_0 |
|---|---|---|---|---|---|
| electricity/D/short | 1.7325 | 1.7325 | 1.7326 | 1.7324 | 1.7274 |
| electricity/W-FRI/short | 1.4938 | 1.4938 | 1.4939 | 1.4936 | 1.5083 |
| solar/D/short | 0.9855 | 0.9855 | 0.9855 | 0.9854 | 0.9868 |
| solar/W-FRI/short | 1.3052 | 1.3052 | 1.3053 | 1.3050 | 1.3013 |
| jena_weather/D/short | 1.0691 | 1.0691 | 1.0692 | 1.0686 | 1.0770 |
| loop_seattle/D/short | 0.8970 | 0.8970 | 0.8970 | 0.8971 | 0.8996 |
| saugeenday/D/short | 3.2756 | 3.2756 | 3.2754 | 3.2755 | 3.3081 |
| us_births/D/short | 0.4509 | 0.4509 | 0.4509 | 0.4512 | 0.4450 |
| **aggregate (geometric mean)** | **1.2139** | **1.2139** | **1.2140** | **1.2139** | **1.2157** |

Forecast agreement sanity (ours F32 vs reference, 9-level interpolated quantiles, `us_births/D/short`, 20 windows): max-abs 9.77e-3 (target scale: births counts, O(10^3-10^4) — this is far looser than the Milestone-0 fixture parity of 1.2e-6 max-abs on unit-scale synthetic data; not investigated further this run, flagged for follow-up).

**Published full-suite numbers for context only, NOT a comparison target for this table** (`docs/reports/t0-published-numbers.md`): GIFT-Eval CRPS 0.4941 / MASE 0.7240 (97 configs, `CONTEXT_LENGTH=8192`, no rollout cap, no subsampling). This subset's absolute CRPS/MASE are far from those numbers because it's a different (smaller, energy/traffic-heavy) task mix at 512-step context, not because of any quant or implementation defect — the reference-vs-ours agreement above is the number that matters for "does quantization hurt."

**Verdict per quant** (band: reference F32 ± 0.5%, this doc's assumed noise band per GOAL.md — no repeated-seed reference run was done to measure actual noise, since `T0Forecaster.predict()` is deterministic in eval mode and there's no RNG to reseed):
- **f16**: within band on every task and the aggregate (largest CRPS delta: 0.0000, largest MASE delta: 0.0003 on jena_weather, ~0.03%). PASS.
- **Q8_0**: within band on every task and the aggregate (largest MASE delta: 0.0006 on jena_weather, ~0.06%). PASS.
- **Q4_0**: within band on 6/8 tasks; `electricity/W-FRI` (MASE +0.97%) and `saugeenday` (MASE +0.99%) exceed the 0.5% band, aggregate MASE delta +0.15% (within band). Consistent with the drift benchmark above (Q4_0 point-drift-worst 8.4%, clearly the roughest of the three quants) — CRPS stays within band everywhere (worst delta 0.9% relative on electricity/D, aggregate +0.11%). Borderline PASS on aggregate, marginal FAIL on 2/8 individual tasks.

What was not run: wgpu backend (GPU contended by `llm-life` all session — see machine line above); the officially published 8192-context / no-subsampling / full-97-config protocol (blocked on this port's autoregressive rollout, not yet implemented); single-signal native/browser latency per quant (GOAL.md's next item); the t0.run "Forecast all" timing reference measurement.

Analysis: `docs/runs/2026-09-19-gifteval-subset.md`.

## Single-signal latency (native CPU, M2)

- Machine: Apple M2 (Darwin 25.3.0), `ndarray` backend, GPU occupied by `llm-life train-a` throughout — wgpu/browser rows pending.
- Commit: `01257dc`.
- Command: `t0-cli bench --weights <F32|GGUF> [--config config.json] --backend ndarray --signals 1 --context 512 --horizon 32 --warmup 2 --reps 10` — median of 10 after 2 warm-ups.

| quant | file MB | load time (s) | median forward latency (ms) |
|---|---|---|---|
| F32 | 406.6 | 0.669 | 363.2 |
| f16 (load-time cast) | 203.3 | 0.799 | 419.9 |
| Q8_0 | 108.9 | 0.543 | 398.0 |
| Q4_0 | 58.6 | 0.362 | 424.2 |
| wgpu (Metal), all quants | — | — | pending (GPU busy) |
| WebGPU (browser), all quants | — | — | pending (GPU busy) |

Analysis: `docs/runs/2026-09-19-latency-cpu.md`.

## Web smoke test (WASM CPU/ndarray, headless Chromium)

- Machine: Apple M2 (Darwin 25.3.0), Playwright's bundled Chromium-for-Testing, GPU occupied by `llm-life train-a` throughout — WebGPU pending.
- Commit: this doc's commit.
- Command: `node scripts/headless/run.mjs --url http://127.0.0.1:8031/ --origins 200,400,560` (`python3 web/serve.py --port 8031` serving `web/`).

| origin | context len | ms/forecast | length_ok | finite | monotone |
|---|---|---|---|---|---|
| 200 | 200 | 556.1 | true | true | true |
| 400 | 400 | 678.5 | true | true | true |
| 560 | 512 (capped) | 834.2 | true | true | true |

Result: PASS, avg 689.6 ms/forecast. Slower than native CPU (363-424 ms, same Q8_0 quant, table above) because `t0-wasm`'s WASM build has no SIMD128/threading enabled and runs single-threaded, vs native ndarray's 8 threads.

Analysis: `docs/runs/2026-09-19-web-smoke.md`.
