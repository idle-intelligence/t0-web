# GIFT-Eval subset — reference F32 vs ours F32/f16/Q8_0/Q4_0 on t0-alpha

Machine: Apple M2 (Darwin 25.3.0). Contention check before every timed run:
`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` — `llm-life`
(another repo's fine-tune) was running throughout this session, so this run
used the `ndarray` (CPU) backend exclusively, never `wgpu`; no wgpu numbers
are reported here.

## Protocol source

- GIFT-Eval: <https://github.com/SalesforceAIResearch/gift-eval> (cloned
  read-only to `/tmp/gift-eval` for inspection; not vendored into this
  repo). Dataset: `Salesforce/GiftEval` on the HF Hub.
- The reference eval procedure is GIFT-Eval's own `notebooks/t0-alpha.ipynb`
  (checked into their repo, so this *is* the official t0-alpha harness, not
  a third-party guess): `gift_eval.data.Dataset` for windowing,
  `gluonts.model.evaluate_model` + `gluonts.ev.metrics.{MASE,
  MeanWeightedSumQuantileLoss}` for scoring, `t0.evaluation.T0Predictor` /
  `T0Forecaster.predict()` for inference.
- CRPS in GIFT-Eval == `MeanWeightedSumQuantileLoss` (mean pinball/quantile
  loss over `QUANTILE_LEVELS`, weighted by the target's absolute sum —
  gluonts' `mean_weighted_sum_quantile_loss`). MASE == seasonal-naive
  normalized MAE, `gluonts.time_feature.get_seasonality(freq)` gives the
  season length (note: gluonts' default map gives seasonality=1 for daily
  and weekly freqs — not 7/52 — this is upstream's convention, reproduced
  as-is, not a bug in this subset).
- `QUANTILE_LEVELS = [0.1, 0.2, ..., 0.9]` (9 levels, the notebook's own
  default) — t0-alpha is trained on 5 levels
  (`config.json`: `[0.1, 0.25, 0.5, 0.75, 0.9]`); the reference model
  interpolates the other 4 internally (`t0.quantile.interpolate_quantiles`,
  piecewise-linear between sorted trained (level, value) pairs, no
  extrapolation needed since 0.1/0.9 are already trained levels). We
  reimplement the identical interpolation in numpy for our own 5-level
  output (`tools/score_gifteval.py::interpolate_quantiles`) — verified
  against `t0.quantile.interpolate_quantiles` to 1.19e-7 max-abs on random
  inputs before trusting any score below.

## Subset (8 configs, term=short only)

Chosen for small download (`fixtures/gifteval/` is 33 MB, well under the
200 MB budget), domain coverage (weather/energy/traffic, per the task, plus
two small clean sanity baselines), and univariate-or-few-variate targets
(so t0-alpha's forward pass, and this port, forecast every series
independently — no multivariate joint-attention path exercised here):

| config | domain | freq | horizon | seasonality | windows | source series |
|---|---|---|---|---|---|---|
| electricity/D/short | Energy | D | 30 | 1 | 1850 | 370 clients × 5 windows |
| electricity/W-FRI/short | Energy | W | 8 | 1 | 1110 | 370 clients × 3 windows |
| solar/D/short | Energy | D | 30 | 1 | 274 | 137 plants × 2 windows |
| solar/W-FRI/short | Energy | W | 8 | 1 | 137 | 137 plants × 1 window |
| jena_weather/D/short | Nature (weather) | D | 30 | 1 | 42 | 21 sensors × 2 windows |
| loop_seattle/D/short | Transport (traffic) | D | 30 | 1 | 646 | 323 loop detectors × 2 windows |
| saugeenday/D/short | Nature (river flow, weather-adjacent) | D | 30 | 1 | 20 | 1 series × 20 windows |
| us_births/D/short | Healthcare | D | 30 | 1 | 20 | 1 series × 20 windows (GIFT-Eval notebook's own quick-demo default) |

4099 windows total. `LOOP_SEATTLE`/`SZ_TAXI`/`M_DENSE` aren't in the
notebook's `dataset_properties.json` domain map (case mismatch), so
`loop_seattle` is the pretty-printed lowercase key used in scored output —
same series, cosmetic only.

Rejected for this subset: everything at sub-hourly frequency (`15T`, `10T`,
`5T`) — `electricity/15T` alone is 208 MB, `LOOP_SEATTLE/5T` 136 MB, both
blow the download budget on their own; `m4_*`/`hospital`/`covid_deaths`
(econ/healthcare, already have two non-weather/energy/traffic baselines);
medium/long terms (task said short-horizon).

## Deviation from the published protocol: context length

The reference notebook uses `CONTEXT_LENGTH = 8192`. This port has no
autoregressive rollout yet (`README.md` "What's not done yet" —
`forecast_series` is one forward pass, capped at `max_horizon=1024` total
sequence length after patch-padding; see
`crates/t0-core/src/model.rs::forecast_series`, fixed this run to support
non-patch-aligned context/horizon via `TimeSeries::pad` + truncate, commit
`03a97eb`). Several of this subset's series (`electricity`, `LOOP_SEATTLE`)
are long enough that a full 8192-context window would need rollout.

**We cap context to the most recent 512 observations for BOTH the
reference and our model, on every window** (`tools/gifteval_subset.py`'s
`CONTEXT_CAP`). This keeps the reference-vs-ours comparison apples-to-apples
— same input, same model architecture, only the weights format differs —
but it means **none of the numbers in `docs/BENCHMARKS.md`'s GIFT-Eval
table are comparable to the officially published full-suite numbers**
(GIFT-Eval CRPS 0.4941 / MASE 0.7240, `docs/reports/t0-published-numbers.md`)
or to a hypothetical 8192-context run of either implementation. They are
only comparable to each other. This is the reason the published numbers
are kept in a separately labelled row in the benchmarks table.

## Pipeline

1. `tools/gifteval_subset.py` (needs `.venv` with `gift_eval`, `gluonts~=0.15.1`,
   `datasets~=2.17.1`, `python-dotenv`; `.env` has `GIFT_EVAL=<local HF snapshot dir>`,
   not committed) — exports each window's trailing-512 context and true
   future to `fixtures/gifteval/*.f32` (gitignored, regenerable) plus
   `fixtures/gifteval/manifest.json` (committed).
2. `t0-cli gifteval --manifest fixtures/gifteval --weights <safetensors|gguf> --out-dir <dir>`
   — one forward pass per window (5 native trained-quantile levels),
   written little-endian f32, window-major/time-major/quantile-major.
3. `tools/score_gifteval.py --which reference|f32|f16|q8_0|q4_0` — reference
   calls `T0Forecaster.predict()` directly per window at the 9 GIFT-Eval
   query levels (its own internal interpolation); ours reads the CLI's
   5-level output and interpolates the same 4 extra levels in numpy. Both
   paths score with the identical `gluonts.model.evaluate_model` call.

Results: `docs/BENCHMARKS.md`. Raw per-task CSVs: `fixtures/gifteval/scores_*.csv` (gitignored).
