# Official-protocol full 97-config — normalized to the GIFT-Eval leaderboard convention

## Parameters

- Input: `results/full97/q4_0/all_results.csv` and `results/full97/f32/all_results.csv`
  (97 rows each, their code/protocol, our dequantized Q4_0 / native F32
  weights — see `docs/runs/2026-09-20-official-protocol-full97.md` for how
  the Q4_0 run was produced; the F32 control was the box run left
  "running" in that doc, finished this update). F32 fetched read-only:
  `scp gpu-box:/path/to/t0-web/results/f32/{all_results.csv,timing_summary.json} results/full97/f32/`.
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

.venv/bin/python3 tools/score_official_protocol.py --variant f32_full97 \
    --results-csv results/full97/f32/all_results.csv \
    --normalize-by results/seasonal_naive/all_results.csv
```

(Reusable for any future weight residency: swap `--results-csv` for that
variant's `all_results.csv`, e.g. `results/full97/q8_0/all_results.csv`
once/if a full-97 Q8_0 box run exists.)

## Release gate table

| variant | normalised MASE | normalised CRPS | vs published card |
|---|---|---|---|
| Published card (t0-alpha, F32) | 0.7240 | 0.4941 | — (`theforecastingcompany/t0-alpha` README lines 319-320) |
| Our F32 (this run, box, their code/protocol) | 0.7255 | 0.4942 | **+0.2% MASE, +0.02% CRPS** |
| Our Q8_0 (full-97) | not available | not available | full-97 Q8_0 was never run on the box this session; only the 8-config subset exists (raw, non-normalizable — see below) |
| Our Q4_0 (this run, box) | 0.7334 | 0.4973 | +1.3% MASE, +0.6% CRPS |

**Gate 1 — does our F32 pipeline reproduce the card:** yes. 0.7255 vs
0.7240 MASE (+0.2%) and 0.4942 vs 0.4941 CRPS (+0.02%) is reproduction to
within normal run-to-run/rounding noise, not a pipeline divergence. This
is the first time this repo has run the full normalized protocol on our
own F32 weights end to end, and it lands on the card almost exactly.

**Gate 2 — Q4_0-vs-F32 quantization cost in isolation (both ours, same
pipeline, same 97 configs):** normalised MASE 0.7334 vs 0.7255
(**+1.09% relative**), normalised CRPS 0.4973 vs 0.4942 (**+0.63%
relative**). This is the number that isolates quantization from any
protocol/pipeline effect, since both rows use our own engine's F32 as the
reference rather than the external card.

**Raw (non-normalized) aggregates**, for reference — not comparable to
the card, included only because the tool prints them: F32 raw MASE
1.0141, CRPS 0.1246; Q4_0 raw MASE 1.0252, CRPS 0.1254 (matches
`docs/runs/2026-09-20-official-protocol-full97.md`'s previously-reported
Q4_0 raw numbers exactly).

## Per-config divergence check (F32 vs Q4_0, not F32 vs card)

The published card gives only the 97-config aggregate, not a per-config
breakdown, so there is no per-config card value to diff F32 against
directly — the aggregate-level check above (Gate 1) is the whole of that
comparison, and it passes. What *is* checkable per config is where Q4_0
moves furthest from our own F32 on the same config, which bounds where
quantization (not a pipeline bug) is doing the most work. Sorted by
`|normalized_Q4_0 - normalized_F32| / normalized_F32`, largest first (all
MASE unless noted):

| config | F32 norm. MASE | Q4_0 norm. MASE | relative diff |
|---|---|---|---|
| bitbrains_fast_storage/H/short | 0.8269 | 0.9279 | +12.2% |
| bitbrains_fast_storage/5T/short | 0.5962 | 0.6655 | +11.6% |
| bitbrains_rnd/5T/short | 0.8352 | 0.9131 | +9.3% |
| bitbrains_fast_storage/5T/long | 0.7970 | 0.8547 | +7.2% |
| bitbrains_fast_storage/5T/medium | 0.8193 | 0.8777 | +7.1% |
| bitbrains_rnd/5T/long | 0.9649 | 1.0196 | +5.7% |
| m4_hourly/H/short | 0.7654 | 0.8072 | +5.5% (CRPS +7.5%, the largest CRPS move) |
| solar/10T/long | 0.9424 | 0.9755 | +3.5% |
| bitbrains_rnd/5T/medium | 0.9721 | 1.0054 | +3.4% |
| bizitobs_l2c/5T/long | 0.4741 | 0.4858 | +2.5% |

All ten of the largest moves are within a low-double-digit percent of
F32 and concentrated in the `bitbrains_*` (VM resource-utilization) and
`m4_hourly` configs — small-magnitude, noisy series where MASE is
naturally more quantization-sensitive, not evidence of a pipeline
divergence (which would show up as a large, systematic shift across many
unrelated configs, not a handful of noisy ones). No config shows F32
itself behaving anomalously against expectation (all F32 normalized
values are in a plausible 0.19-1.19 range, consistent with the rest of
the table below); the aggregate-level Gate 1 match is corroborated rather
than contradicted at the per-config level.

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

## Per-config normalised table (F32 and Q4_0 full-97, from `--normalize-by`)

| config | F32 norm. MASE | F32 norm. CRPS | Q4_0 norm. MASE | Q4_0 norm. CRPS |
|---|---|---|---|---|
| loop_seattle/5T/short | 0.7421 | 0.5948 | 0.7400 | 0.5932 |
| loop_seattle/5T/medium | 0.6670 | 0.5794 | 0.6689 | 0.5816 |
| loop_seattle/5T/long | 0.6490 | 0.5647 | 0.6504 | 0.5662 |
| loop_seattle/D/short | 0.5133 | 0.4198 | 0.5141 | 0.4203 |
| loop_seattle/H/short | 0.6316 | 0.5461 | 0.6355 | 0.5493 |
| loop_seattle/H/medium | 0.5982 | 0.3738 | 0.6016 | 0.3760 |
| loop_seattle/H/long | 0.5664 | 0.3168 | 0.5683 | 0.3175 |
| m_dense/D/short | 0.4122 | 0.3008 | 0.4122 | 0.3019 |
| m_dense/H/short | 0.5455 | 0.4882 | 0.5489 | 0.4918 |
| m_dense/H/medium | 0.4346 | 0.3005 | 0.4381 | 0.3026 |
| m_dense/H/long | 0.4588 | 0.2689 | 0.4634 | 0.2716 |
| sz_taxi/15T/short | 0.7136 | 0.6502 | 0.7137 | 0.6500 |
| sz_taxi/15T/medium | 0.7465 | 0.5275 | 0.7462 | 0.5272 |
| sz_taxi/15T/long | 0.7246 | 0.4533 | 0.7253 | 0.4534 |
| sz_taxi/H/short | 0.7592 | 0.6332 | 0.7590 | 0.6331 |
| bitbrains_fast_storage/5T/short | 0.5962 | 0.3124 | 0.6655 | 0.3171 |
| bitbrains_fast_storage/5T/medium | 0.8193 | 0.4833 | 0.8777 | 0.4823 |
| bitbrains_fast_storage/5T/long | 0.7970 | 0.5393 | 0.8547 | 0.5388 |
| bitbrains_fast_storage/H/short | 0.8269 | 0.5751 | 0.9279 | 0.5728 |
| bitbrains_rnd/5T/short | 0.8352 | 0.3643 | 0.9131 | 0.3657 |
| bitbrains_rnd/5T/medium | 0.9721 | 0.5306 | 1.0054 | 0.5264 |
| bitbrains_rnd/5T/long | 0.9649 | 0.5059 | 1.0196 | 0.4981 |
| bitbrains_rnd/H/short | 0.9597 | 0.4675 | 0.9856 | 0.4772 |
| bizitobs_application/10S/short | 0.5671 | 0.3492 | 0.5588 | 0.3468 |
| bizitobs_application/10S/medium | 0.9157 | 0.8841 | 0.9237 | 0.9009 |
| bizitobs_application/10S/long | 0.9887 | 1.1153 | 0.9941 | 1.1290 |
| bizitobs_l2c/5T/short | 0.2727 | 0.2708 | 0.2735 | 0.2730 |
| bizitobs_l2c/5T/medium | 0.4937 | 0.4751 | 0.5038 | 0.4895 |
| bizitobs_l2c/5T/long | 0.4741 | 0.4690 | 0.4858 | 0.4868 |
| bizitobs_l2c/H/short | 0.3932 | 0.3773 | 0.3994 | 0.3809 |
| bizitobs_l2c/H/medium | 0.3386 | 0.2694 | 0.3425 | 0.2716 |
| bizitobs_l2c/H/long | 0.3934 | 0.2750 | 0.3973 | 0.2776 |
| bizitobs_service/10S/short | 0.6706 | 0.3181 | 0.6862 | 0.3252 |
| bizitobs_service/10S/medium | 0.8816 | 0.5098 | 0.8891 | 0.5088 |
| bizitobs_service/10S/long | 1.0626 | 0.9625 | 1.0580 | 0.9473 |
| car_parts/M/short | 0.7036 | 0.5610 | 0.7050 | 0.5606 |
| covid_deaths/D/short | 0.8864 | 0.4889 | 0.8960 | 0.4875 |
| electricity/15T/short | 0.6317 | 0.5774 | 0.6352 | 0.5789 |
| electricity/15T/medium | 0.7434 | 0.6765 | 0.7464 | 0.6800 |
| electricity/15T/long | 0.7772 | 0.6917 | 0.7798 | 0.6934 |
| electricity/D/short | 0.6988 | 0.5182 | 0.6995 | 0.5180 |
| electricity/H/short | 0.7000 | 0.6315 | 0.7006 | 0.6339 |
| electricity/H/medium | 0.7619 | 0.5596 | 0.7654 | 0.5657 |
| electricity/H/long | 0.7486 | 0.5015 | 0.7545 | 0.5095 |
| electricity/W/short | 0.7267 | 0.5727 | 0.7327 | 0.5793 |
| ett1/15T/short | 0.7463 | 0.6764 | 0.7458 | 0.6777 |
| ett1/15T/medium | 0.8763 | 0.7630 | 0.8813 | 0.7710 |
| ett1/15T/long | 0.9210 | 0.7483 | 0.9239 | 0.7586 |
| ett1/D/short | 1.0440 | 0.7396 | 1.0403 | 0.7355 |
| ett1/H/short | 0.8912 | 0.7969 | 0.8944 | 0.7991 |
| ett1/H/medium | 0.9425 | 0.6829 | 0.9445 | 0.6898 |
| ett1/H/long | 1.0110 | 0.6435 | 1.0148 | 0.6495 |
| ett1/W/short | 0.8893 | 0.8367 | 0.8777 | 0.8226 |
| ett2/15T/short | 0.6886 | 0.6615 | 0.6870 | 0.6578 |
| ett2/15T/medium | 0.8472 | 0.7329 | 0.8456 | 0.7323 |
| ett2/15T/long | 0.9033 | 0.7081 | 0.9037 | 0.7084 |
| ett2/D/short | 1.0148 | 0.6352 | 1.0044 | 0.6334 |
| ett2/H/short | 0.7758 | 0.6948 | 0.7802 | 0.6996 |
| ett2/H/medium | 0.8213 | 0.5367 | 0.8205 | 0.5373 |
| ett2/H/long | 0.9407 | 0.5102 | 0.9363 | 0.5091 |
| ett2/W/short | 1.1659 | 0.6636 | 1.1879 | 0.6702 |
| hierarchical_sales/D/short | 0.6556 | 0.3291 | 0.6554 | 0.3287 |
| hierarchical_sales/W/short | 0.7131 | 0.4259 | 0.7126 | 0.4261 |
| hospital/M/short | 0.8485 | 0.8925 | 0.8474 | 0.8869 |
| jena_weather/10T/short | 0.3626 | 0.1755 | 0.3715 | 0.1793 |
| jena_weather/10T/medium | 0.8316 | 0.2216 | 0.8402 | 0.2238 |
| jena_weather/10T/long | 0.8652 | 0.2035 | 0.8699 | 0.2054 |
| jena_weather/D/short | 0.6795 | 0.2173 | 0.6845 | 0.2172 |
| jena_weather/H/short | 0.7415 | 0.2694 | 0.7472 | 0.2755 |
| jena_weather/H/medium | 0.8622 | 0.1489 | 0.8701 | 0.1539 |
| jena_weather/H/long | 0.6540 | 0.1260 | 0.6650 | 0.1293 |
| kdd_cup_2018/D/short | 0.8139 | 0.5700 | 0.8172 | 0.5723 |
| kdd_cup_2018/H/short | 0.6804 | 0.6654 | 0.6812 | 0.6648 |
| kdd_cup_2018/H/medium | 0.7119 | 0.5424 | 0.7109 | 0.5414 |
| kdd_cup_2018/H/long | 0.7469 | 0.4654 | 0.7477 | 0.4652 |
| m4_daily/D/short | 1.0421 | 0.9095 | 1.0342 | 0.9024 |
| m4_hourly/H/short | 0.7654 | 0.5486 | 0.8072 | 0.5897 |
| m4_monthly/M/short | 0.7818 | 0.7865 | 0.7782 | 0.7842 |
| m4_quarterly/Q/short | 0.8190 | 0.8155 | 0.8143 | 0.8140 |
| m4_weekly/W/short | 0.8213 | 0.6577 | 0.8462 | 0.6671 |
| m4_yearly/A/short | 1.0006 | 0.9637 | 0.9937 | 0.9582 |
| restaurant/D/short | 0.6693 | 0.3739 | 0.6697 | 0.3740 |
| saugeen/D/short | 0.8774 | 0.6025 | 0.8845 | 0.6048 |
| saugeen/M/short | 0.7509 | 0.6489 | 0.7530 | 0.6523 |
| saugeen/W/short | 0.5646 | 0.4540 | 0.5640 | 0.4553 |
| solar/10T/short | 0.8513 | 0.5719 | 0.8711 | 0.5803 |
| solar/10T/medium | 0.9166 | 0.5153 | 0.9350 | 0.5208 |
| solar/10T/long | 0.9424 | 0.4689 | 0.9755 | 0.4855 |
| solar/D/short | 0.8543 | 0.4924 | 0.8559 | 0.4914 |
| solar/H/short | 0.9837 | 0.5751 | 0.9969 | 0.5888 |
| solar/H/medium | 1.0792 | 0.3807 | 1.0726 | 0.3863 |
| solar/H/long | 0.9558 | 0.3193 | 0.9585 | 0.3258 |
| solar/W/short | 0.8759 | 0.8946 | 0.8730 | 0.8924 |
| temperature_rain/D/short | 0.6829 | 0.4455 | 0.6874 | 0.4458 |
| us_births/D/short | 0.1838 | 0.1520 | 0.1864 | 0.1536 |
| us_births/M/short | 0.6456 | 0.6786 | 0.6586 | 0.6885 |
| us_births/W/short | 0.7457 | 0.6961 | 0.7411 | 0.6937 |


`us_births/D/short` (normalized MASE 0.1864) is a strong outlier — seasonal
naive does very poorly on that config in absolute terms (consistent with
this repo's own raw table showing `us_births/D/short` MASE 0.34-0.35 across
F32/Q8_0/Q4_0 in `docs/runs/2026-09-20-official-protocol-fast.md`), which is
a property of the baseline on that dataset, not a t0-alpha-specific
anomaly — noted here rather than excluded, since the leaderboard formula
takes all 97 configs as-is.
