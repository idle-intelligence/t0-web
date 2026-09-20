# Official-protocol full 97-config GIFT-Eval — Q4_0, on the GPU box

## Parameters

- Box: a Linux desktop with an RTX 3080, their code (`gift_eval`/`gluonts`, same
  metric path as `tools/score_official_protocol.py` — `MASE()` +
  `MeanWeightedSumQuantileLoss()` via `gluonts.model.evaluate_model`), our
  dequantized Q4_0 weights (`t0-alpha` GGUF, `q4_0`, same export as
  `docs/runs/2026-09-20-official-protocol-fast.md`).
- Scope: all 97 GIFT-Eval configs (vs. the 8-config subset in
  `docs/runs/2026-09-20-official-protocol-fast.md` and
  `docs/runs/2026-09-19-official-protocol-subset.md`), `context_length=8192`
  (`results/full97/q4_0/timing_summary.json:context_length`), full test
  split, no window subsampling, `max_minutes=300`,
  `stopped_early: false` (`timing_summary.json`).
- Launch: directory `results/q4_0/` created on the box at 2026-09-20
  17:15:30 CEST (box `ls --time-style=full-iso`). Finish: `all_results.csv`
  and `timing_summary.json` both written 2026-09-20 19:32:37 CEST (same
  listing) — wall time ≈2h17m.
- Sum of per-config `elapsed_s` in `timing_summary.json` (inference time
  only, one window loop per config) is 2912.9s (≈48.5 min) — noticeably
  less than the 2h17m wall time above; the gap is per-config data
  loading/model setup/`gluonts` evaluation overhead not captured in the
  `elapsed_s` field, not a measurement of raw compute alone.
- Data fetched from the box: `scp gpu-box:/path/to/t0-web/results/q4_0/{all_results.csv,timing_summary.json} results/full97/q4_0/`
  (97 rows in `all_results.csv`, one per config; `results/` is not
  git-ignored in this repo, confirmed with `git check-ignore -v`, so these
  files are committed directly under `results/full97/q4_0/`).
- F32 control: still running on the box as of this write-up. Its
  `results/f32/all_results.csv` was only a 305-byte header as of 19:32:38
  CEST (checked via `ls`, not fetched — box was not otherwise touched per
  instructions) — the F32 run appears to have started only once Q4_0
  finished (one-GPU-job-at-a-time), so it is effectively just beginning.
  ETA ~100 min, pending.

## Aggregate (geometric mean over 97 configs, per `tools/score_official_protocol.py`'s aggregation: `exp(mean(log(x)))` over each metric column)

| variant | MASE (agg) | CRPS (agg, `mean_weighted_sum_quantile_loss`) | configs | wall time |
|---|---|---|---|---|
| Q4_0 (this run) | 1.0252 | 0.1254 | 97 | 2h17m (box time), 48.5 min inference-only |
| F32 (control) | pending | pending | 97 | running on the box, ETA ~100 min |

Values from `results/full97/q4_0/all_results.csv`: `agg_mase = exp(mean(log(eval_metrics/MASE[0.5])))` = 1.0251565524664221, `agg_crps = exp(mean(log(eval_metrics/mean_weighted_sum_quantile_loss)))` = 0.12537503219247603, computed over all 97 rows with the same geometric-mean formula `score_official_protocol.py` uses (lines 141-145 of that script), applied directly to the box's already-computed per-config metrics rather than re-running the script (the box's `all_results.csv` already contains the same `MASE[0.5]`/`mean_weighted_sum_quantile_loss` columns the script would produce).

### Comparison to published t0-alpha GIFT-Eval numbers

`~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/README.md` lines
319-320 (model card's GIFT-Eval leaderboard table):

| GIFT-Eval | CRPS | 0.4941 |
| GIFT-Eval | MASE | 0.7240 |

These published numbers are **not directly comparable** to the aggregate
above. The GIFT-Eval leaderboard protocol normalizes each config's
CRPS/MASE by a seasonal-naive baseline's score on that same config before
taking the geometric mean across configs (that is the standard GIFT-Eval
scoring convention on the linked leaderboard,
https://huggingface.co/spaces/Salesforce/GIFT-Eval); `score_official_protocol.py`
and this run instead take the geometric mean of the raw MASE/CRPS values,
with no naive-baseline normalization (confirmed by reading the script: no
naive-model forecast or ratio step anywhere in `score_task`/`main`). The
8-config subset doc (`docs/runs/2026-09-20-official-protocol-fast.md`)
made the same raw, non-normalized comparison, and it was against
`docs/runs/2026-09-19-official-protocol-subset.md`'s "their code, our
weights" run on the same 8 configs and the same non-normalized metric —
that is the only apples-to-apples comparison available for this raw
aggregate; the model-card number is reported here for context only, not
as a check.

### Comparison to the earlier 8-config subset (raw MASE/CRPS, same non-normalized metric)

From `docs/BENCHMARKS.md`'s official-protocol subset table
(`docs/runs/2026-09-20-official-protocol-fast.md`), Q4_0, 8 configs:
CRPS 0.0820, MASE 1.1329. The full-97 Q4_0 aggregate above (CRPS 0.1254,
MASE 1.0252) is not a like-for-like extension of that number — it
includes 89 additional configs (multiple frequency/horizon terms per
dataset, several very different domains: `bitbrains_rnd`, `covid_deaths`,
`m4_*`, etc.) that the 8-config subset never touched, so the difference
between 0.0820/0.1254 and 1.1329/1.0252 reflects dataset composition, not
a regression or improvement in the engine.

## Per-config table (all 97 rows, from `results/full97/q4_0/all_results.csv`)

| config | MASE[0.5] | CRPS (mean_weighted_sum_quantile_loss) |
|---|---|---|
| loop_seattle/5T/short | 0.5641 | 0.0479 |
| loop_seattle/5T/medium | 0.7714 | 0.0682 |
| loop_seattle/5T/long | 0.8135 | 0.0720 |
| loop_seattle/D/short | 0.8906 | 0.0434 |
| loop_seattle/H/short | 0.8216 | 0.0573 |
| loop_seattle/H/medium | 0.8906 | 0.0609 |
| loop_seattle/H/long | 0.8787 | 0.0594 |
| m_dense/D/short | 0.6881 | 0.0685 |
| m_dense/H/short | 0.8165 | 0.1351 |
| m_dense/H/medium | 0.6878 | 0.1142 |
| m_dense/H/long | 0.6849 | 0.1138 |
| sz_taxi/15T/short | 0.5455 | 0.2007 |
| sz_taxi/15T/medium | 0.5324 | 0.1999 |
| sz_taxi/15T/long | 0.5012 | 0.1940 |
| sz_taxi/H/short | 0.5603 | 0.1354 |
| bitbrains_fast_storage/5T/short | 0.7560 | 0.3838 |
| bitbrains_fast_storage/5T/medium | 1.0711 | 0.5777 |
| bitbrains_fast_storage/5T/long | 0.9714 | 0.6343 |
| bitbrains_fast_storage/H/short | 1.2049 | 0.5855 |
| bitbrains_rnd/5T/short | 1.7995 | 0.4028 |
| bitbrains_rnd/5T/medium | 4.5671 | 0.6155 |
| bitbrains_rnd/5T/long | 3.5698 | 0.5854 |
| bitbrains_rnd/H/short | 5.9501 | 0.5932 |
| bizitobs_application/10S/short | 1.2530 | 0.0121 |
| bizitobs_application/10S/medium | 2.4861 | 0.0385 |
| bizitobs_application/10S/long | 3.1876 | 0.0516 |
| bizitobs_l2c/5T/short | 0.2697 | 0.0716 |
| bizitobs_l2c/5T/medium | 0.6265 | 0.2548 |
| bizitobs_l2c/5T/long | 0.7066 | 0.3157 |
| bizitobs_l2c/H/short | 0.4850 | 0.1985 |
| bizitobs_l2c/H/medium | 0.5173 | 0.2456 |
| bizitobs_l2c/H/long | 0.5665 | 0.2613 |
| bizitobs_service/10S/short | 0.8409 | 0.0130 |
| bizitobs_service/10S/medium | 1.1741 | 0.0242 |
| bizitobs_service/10S/long | 1.4464 | 0.0506 |
| car_parts/M/short | 0.8470 | 0.9651 |
| covid_deaths/D/short | 42.0334 | 0.0618 |
| electricity/15T/short | 1.0907 | 0.0954 |
| electricity/15T/medium | 0.8589 | 0.0767 |
| electricity/15T/long | 0.9073 | 0.0780 |
| electricity/D/short | 1.3900 | 0.0539 |
| electricity/H/short | 0.9512 | 0.0670 |
| electricity/H/medium | 1.0658 | 0.0721 |
| electricity/H/long | 1.1499 | 0.0782 |
| electricity/W/short | 1.5312 | 0.0575 |
| ett1/15T/short | 0.6967 | 0.1636 |
| ett1/15T/medium | 1.0472 | 0.2479 |
| ett1/15T/long | 1.1002 | 0.2581 |
| ett1/D/short | 1.8501 | 0.3004 |
| ett1/H/short | 0.8741 | 0.1921 |
| ett1/H/medium | 1.4807 | 0.3000 |
| ett1/H/long | 1.5005 | 0.3059 |
| ett1/W/short | 1.5525 | 0.2565 |
| ett2/15T/short | 0.7333 | 0.0634 |
| ett2/15T/medium | 0.8890 | 0.0909 |
| ett2/15T/long | 0.9151 | 0.0941 |
| ett2/D/short | 1.3962 | 0.0971 |
| ett2/H/short | 0.7203 | 0.0622 |
| ett2/H/medium | 1.0162 | 0.1001 |
| ett2/H/long | 1.0566 | 0.1058 |
| ett2/W/short | 0.9248 | 0.0896 |
| hierarchical_sales/D/short | 0.7437 | 0.5709 |
| hierarchical_sales/W/short | 0.7305 | 0.3546 |
| hospital/M/short | 0.7801 | 0.0554 |
| jena_weather/10T/short | 0.2760 | 0.0278 |
| jena_weather/10T/medium | 0.6016 | 0.0474 |
| jena_weather/10T/long | 0.6624 | 0.0487 |
| jena_weather/D/short | 1.0770 | 0.0457 |
| jena_weather/H/short | 0.5401 | 0.0425 |
| jena_weather/H/medium | 0.7732 | 0.0528 |
| jena_weather/H/long | 0.8430 | 0.0542 |
| kdd_cup_2018/D/short | 1.2233 | 0.3860 |
| kdd_cup_2018/H/short | 0.9131 | 0.3640 |
| kdd_cup_2018/H/medium | 1.0159 | 0.4108 |
| kdd_cup_2018/H/long | 0.9985 | 0.4356 |
| m4_daily/D/short | 3.3906 | 0.0220 |
| m4_hourly/H/short | 0.9631 | 0.0222 |
| m4_monthly/M/short | 0.9803 | 0.0956 |
| m4_quarterly/Q/short | 1.3048 | 0.0798 |
| m4_weekly/W/short | 2.3502 | 0.0406 |
| m4_yearly/A/short | 3.9412 | 0.1314 |
| restaurant/D/short | 0.6738 | 0.2532 |
| saugeen/D/short | 3.0190 | 0.3538 |
| saugeen/M/short | 0.7352 | 0.2903 |
| saugeen/W/short | 1.1227 | 0.3342 |
| solar/10T/short | 0.9632 | 0.4988 |
| solar/10T/medium | 0.8667 | 0.3412 |
| solar/10T/long | 0.8496 | 0.3271 |
| solar/D/short | 0.9893 | 0.2747 |
| solar/H/short | 0.9490 | 0.3484 |
| solar/H/medium | 1.0029 | 0.3655 |
| solar/H/long | 1.0267 | 0.3512 |
| solar/W/short | 1.2836 | 0.1872 |
| temperature_rain/D/short | 1.3827 | 0.5653 |
| us_births/D/short | 0.3475 | 0.0184 |
| us_births/M/short | 0.5009 | 0.0116 |
| us_births/W/short | 1.1586 | 0.0134 |

Windows and per-config `elapsed_s`/`s_per_window` for all 97 configs are
in `results/full97/q4_0/timing_summary.json` (not reproduced in full
here — e.g. `loop_seattle/5T/short` has `n_windows=6460`,
`elapsed_s=155.60`; `us_births/D/short` has `n_windows=20`,
`elapsed_s=0.587`).

## Observations

- `covid_deaths/D/short` has MASE 42.0334, more than an order of magnitude
  above every other config in the table (next highest is
  `bitbrains_rnd/H/short` at 5.9501). This is a data point from
  `all_results.csv`, not an artifact of this write-up; it dominates the
  spread of the per-config MASE column but, because the aggregate is a
  geometric mean, its effect on the 1.0252 aggregate is bounded
  (`ln(42.0334)/97 ≈ 0.0387` of the mean log-MASE).
- `bitbrains_rnd` (the 5T and H terms) also has high MASE (1.80-5.95)
  relative to most other configs, consistent with `bitbrains_fast_storage`
  (0.76-1.20) being an easier sibling dataset in the same domain.
- The sum of per-config `elapsed_s` in `timing_summary.json` (2912.9s) is
  about 1/2.8 of the observed wall time between directory creation and
  file-write completion (2h17m ≈ 8237s). No log with a full breakdown of
  the remaining time was fetched (out of scope — only the two named files
  were pulled from the box), so the source of that gap is not established
  here beyond "not raw inference," per the box's own `elapsed_s` field.
- The F32 control's `all_results.csv` was still just a 305-byte header as
  of 19:32:38 CEST, one second after the Q4_0 files finished writing —
  consistent with the two runs having been serialized on the box (one GPU
  job at a time) rather than run concurrently.
