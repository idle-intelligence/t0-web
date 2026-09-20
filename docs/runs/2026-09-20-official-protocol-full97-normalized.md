# Official-protocol full 97-config — normalized to the GIFT-Eval leaderboard convention

## Parameters

- Input: `results/full97/q4_0/all_results.csv` (97 rows, Q4_0, their code/protocol,
  our dequantized Q4_0 weights — see `docs/runs/2026-09-20-official-protocol-full97.md`
  for how it was produced).
- Baseline: Seasonal Naive per-config results, fetched read-only from
  `github.com/SalesforceAIResearch/gift-eval`, commit `9a014e9e8ea130ba39c100c60d5dcbab7db57ac9`,
  `results/seasonal_naive/all_results.csv` (`git clone --depth 1` into
  `$CLAUDE_JOB_DIR/tmp/gift-eval`, then copied verbatim to
  `results/seasonal_naive/all_results.csv` in this repo, 97 rows + header).
- Definition, with citation: the GIFT-Eval leaderboard's normalization is
  in `notebooks/zeus.ipynb`'s "Print Results" cell (same commit as above).
  Per config (joined on the `dataset` column, e.g. `"loop_seattle/5T/short"`):
  `normalized_MASE = MASE / seasonal_naive_MASE`, `normalized_CRPS = CRPS /
  seasonal_naive_CRPS`. The aggregate is `geo_mean(x) = prod(x)**(1/n)`
  over the 97 configs, applied separately to the normalized MASE and CRPS
  vectors — algebraically identical to this repo's own `exp(mean(log(x)))`
  aggregate, just applied to the ratios instead of the raw values.
- Join check: all 97 `dataset` keys in `results/full97/q4_0/all_results.csv`
  matched exactly against `results/seasonal_naive/all_results.csv`'s 97
  keys (both are gift-eval's native `name/freq/term` format, e.g.
  `"loop_seattle/5T/short"`) — **0 unmatched configs**, no naming
  reconciliation was needed for this run.
- Tool: `tools/score_official_protocol.py` gained `--results-csv` (score an
  already-computed `all_results.csv` directly, since the box run's
  manifest/forecast files were regenerable `/tmp` data, not committed — see
  below) and `--normalize-by <seasonal_naive_csv>` (this normalization,
  with the unmatched-key check above built in and failing loudly on any
  gap).

## Command

```
.venv/bin/python3 tools/score_official_protocol.py --variant q4_0_full97 \
    --results-csv results/full97/q4_0/all_results.csv \
    --normalize-by results/seasonal_naive/all_results.csv
```

F32 control (pending — reuse the same command once `results/full97/f32/all_results.csv` exists):

```
.venv/bin/python3 tools/score_official_protocol.py --variant f32_full97 \
    --results-csv results/full97/f32/all_results.csv \
    --normalize-by results/seasonal_naive/all_results.csv
```

## Results

| run | raw MASE | raw CRPS | normalised MASE | normalised CRPS | published card (t0-alpha, F32) |
|---|---|---|---|---|---|
| Q4_0 full-97 (this run) | 1.0252 | 0.1254 | 0.7334 | 0.4973 | MASE 0.7240, CRPS 0.4941 (`theforecastingcompany/t0-alpha` README lines 319-320) |
| F32 full-97 | pending | pending | pending | pending | — |

The normalised Q4_0 aggregate (MASE 0.7334, CRPS 0.4973) is the number
directly comparable to the published card: **+1.3% MASE, +0.6% CRPS**
relative to the card's F32 numbers — consistent with a small, expected
lossy-quantization gap from Q4_0 rather than a protocol mismatch, given
that the raw (non-normalized) aggregate above (1.0252 / 0.1254) has no
meaningful relationship to the card's numbers on its own (different
scale entirely — see `docs/runs/2026-09-20-official-protocol-full97.md`'s
normalization caveat, now resolved by this run). Both the card's F32
number and this Q4_0 number are on the same 97-config set and the same
official protocol (`gluonts.model.evaluate_model`, full test split, no
subsampling); the residual difference is attributable to the model card
being F32 and this run being Q4_0 — the pending F32 control above would
isolate that gap directly once available.

## 8-config subset (F32/Q8_0/Q4_0) — not available

`docs/runs/2026-09-20-official-protocol-fast.md` (our `t0-fast` engine
on the same 8 configs) explicitly documents that its manifest,
per-window context/future `.f32` files, and forecast outputs were
"regenerable data ... written to `/tmp` this session rather than
committed" (that doc's line under "Commands"). Checked this session:
none of `/tmp/gifteval_official*` exist anymore, and no per-config
`all_results.csv`-style file for that subset was ever committed under
`results/`. Only the aggregate numbers survive, in that doc's tables and
in `docs/BENCHMARKS.md`'s official-protocol subset table — those are
already the raw (non-normalized) metric, and cannot be renormalized here
without the underlying per-config forecasts, which would require
re-running `t0-cli gifteval --backend fast` (a GPU job, out of scope for
this session). If that subset is regenerated in a future session, the
same `--results-csv`/`--normalize-by` flags added here apply directly —
the 8 configs are also gift-eval's `name/freq/term` keys and are a subset
of `results/seasonal_naive/all_results.csv`'s 97 rows.

## Per-config normalised table (Q4_0 full-97, from `--normalize-by`)

| config | normalized MASE | normalized CRPS |
|---|---|---|
| loop_seattle/5T/short | 0.7400 | 0.5932 |
| loop_seattle/5T/medium | 0.6689 | 0.5816 |
| loop_seattle/5T/long | 0.6504 | 0.5662 |
| loop_seattle/D/short | 0.5141 | 0.4203 |
| loop_seattle/H/short | 0.6355 | 0.5493 |
| loop_seattle/H/medium | 0.6016 | 0.3760 |
| loop_seattle/H/long | 0.5683 | 0.3175 |
| m_dense/D/short | 0.4122 | 0.3019 |
| m_dense/H/short | 0.5489 | 0.4918 |
| m_dense/H/medium | 0.4381 | 0.3026 |
| m_dense/H/long | 0.4634 | 0.2716 |
| sz_taxi/15T/short | 0.7137 | 0.6500 |
| sz_taxi/15T/medium | 0.7462 | 0.5272 |
| sz_taxi/15T/long | 0.7253 | 0.4534 |
| sz_taxi/H/short | 0.7590 | 0.6331 |
| bitbrains_fast_storage/5T/short | 0.6655 | 0.3171 |
| bitbrains_fast_storage/5T/medium | 0.8777 | 0.4823 |
| bitbrains_fast_storage/5T/long | 0.8547 | 0.5388 |
| bitbrains_fast_storage/H/short | 0.9279 | 0.5728 |
| bitbrains_rnd/5T/short | 0.9131 | 0.3657 |
| bitbrains_rnd/5T/medium | 1.0054 | 0.5264 |
| bitbrains_rnd/5T/long | 1.0196 | 0.4981 |
| bitbrains_rnd/H/short | 0.9856 | 0.4772 |
| bizitobs_application/10S/short | 0.5588 | 0.3468 |
| bizitobs_application/10S/medium | 0.9237 | 0.9009 |
| bizitobs_application/10S/long | 0.9941 | 1.1290 |
| bizitobs_l2c/5T/short | 0.2735 | 0.2730 |
| bizitobs_l2c/5T/medium | 0.5038 | 0.4895 |
| bizitobs_l2c/5T/long | 0.4858 | 0.4868 |
| bizitobs_l2c/H/short | 0.3994 | 0.3809 |
| bizitobs_l2c/H/medium | 0.3425 | 0.2716 |
| bizitobs_l2c/H/long | 0.3973 | 0.2776 |
| bizitobs_service/10S/short | 0.6862 | 0.3252 |
| bizitobs_service/10S/medium | 0.8891 | 0.5088 |
| bizitobs_service/10S/long | 1.0580 | 0.9473 |
| car_parts/M/short | 0.7050 | 0.5606 |
| covid_deaths/D/short | 0.8960 | 0.4875 |
| electricity/15T/short | 0.6352 | 0.5789 |
| electricity/15T/medium | 0.7464 | 0.6800 |
| electricity/15T/long | 0.7798 | 0.6934 |
| electricity/D/short | 0.6995 | 0.5180 |
| electricity/H/short | 0.7006 | 0.6339 |
| electricity/H/medium | 0.7654 | 0.5657 |
| electricity/H/long | 0.7545 | 0.5095 |
| electricity/W/short | 0.7327 | 0.5793 |
| ett1/15T/short | 0.7458 | 0.6777 |
| ett1/15T/medium | 0.8813 | 0.7710 |
| ett1/15T/long | 0.9239 | 0.7586 |
| ett1/D/short | 1.0403 | 0.7355 |
| ett1/H/short | 0.8944 | 0.7991 |
| ett1/H/medium | 0.9445 | 0.6898 |
| ett1/H/long | 1.0148 | 0.6495 |
| ett1/W/short | 0.8777 | 0.8226 |
| ett2/15T/short | 0.6870 | 0.6578 |
| ett2/15T/medium | 0.8456 | 0.7323 |
| ett2/15T/long | 0.9037 | 0.7084 |
| ett2/D/short | 1.0044 | 0.6334 |
| ett2/H/short | 0.7802 | 0.6996 |
| ett2/H/medium | 0.8205 | 0.5373 |
| ett2/H/long | 0.9363 | 0.5091 |
| ett2/W/short | 1.1879 | 0.6702 |
| hierarchical_sales/D/short | 0.6554 | 0.3287 |
| hierarchical_sales/W/short | 0.7126 | 0.4261 |
| hospital/M/short | 0.8474 | 0.8869 |
| jena_weather/10T/short | 0.3715 | 0.1793 |
| jena_weather/10T/medium | 0.8402 | 0.2238 |
| jena_weather/10T/long | 0.8699 | 0.2054 |
| jena_weather/D/short | 0.6845 | 0.2172 |
| jena_weather/H/short | 0.7472 | 0.2755 |
| jena_weather/H/medium | 0.8701 | 0.1539 |
| jena_weather/H/long | 0.6650 | 0.1293 |
| kdd_cup_2018/D/short | 0.8172 | 0.5723 |
| kdd_cup_2018/H/short | 0.6812 | 0.6648 |
| kdd_cup_2018/H/medium | 0.7109 | 0.5414 |
| kdd_cup_2018/H/long | 0.7477 | 0.4652 |
| m4_daily/D/short | 1.0342 | 0.9024 |
| m4_hourly/H/short | 0.8072 | 0.5897 |
| m4_monthly/M/short | 0.7782 | 0.7842 |
| m4_quarterly/Q/short | 0.8143 | 0.8140 |
| m4_weekly/W/short | 0.8462 | 0.6671 |
| m4_yearly/A/short | 0.9937 | 0.9582 |
| restaurant/D/short | 0.6697 | 0.3740 |
| saugeen/D/short | 0.8845 | 0.6048 |
| saugeen/M/short | 0.7530 | 0.6523 |
| saugeen/W/short | 0.5640 | 0.4553 |
| solar/10T/short | 0.8711 | 0.5803 |
| solar/10T/medium | 0.9350 | 0.5208 |
| solar/10T/long | 0.9755 | 0.4855 |
| solar/D/short | 0.8559 | 0.4914 |
| solar/H/short | 0.9969 | 0.5888 |
| solar/H/medium | 1.0726 | 0.3863 |
| solar/H/long | 0.9585 | 0.3258 |
| solar/W/short | 0.8730 | 0.8924 |
| temperature_rain/D/short | 0.6874 | 0.4458 |
| us_births/D/short | 0.1864 | 0.1536 |
| us_births/M/short | 0.6586 | 0.6885 |
| us_births/W/short | 0.7411 | 0.6937 |

`us_births/D/short` (normalized MASE 0.1864) is a strong outlier — seasonal
naive does very poorly on that config in absolute terms (consistent with
this repo's own raw table showing `us_births/D/short` MASE 0.34-0.35 across
F32/Q8_0/Q4_0 in `docs/runs/2026-09-20-official-protocol-fast.md`), which is
a property of the baseline on that dataset, not a t0-alpha-specific
anomaly — noted here rather than excluded, since the leaderboard formula
takes all 97 configs as-is.
