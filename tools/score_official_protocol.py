"""Score `t0-cli gifteval --backend fast`'s output (per
`tools/gifteval_official_export.py`'s manifest -- full test split, no
subsampling, ~8192-context windows) with the SAME gluonts metric code as
`tools/score_gifteval.py` (`gluonts.model.evaluate_model` +
`MASE`/`MeanWeightedSumQuantileLoss`), so the numbers are comparable to
`docs/runs/2026-09-19-official-protocol-subset.md`'s "their code, their
protocol" table (that doc never touches our Rust/Burn port; this script is
the missing "ours, same protocol" half).

Interpolation from the 5 trained quantile levels to GIFT-Eval's 9 query
levels is the same piecewise-linear reimplementation `tools/score_gifteval.py`
already carries and has verified against `t0.quantile.interpolate_quantiles`.

Usage:
    .venv/bin/python3 tools/score_official_protocol.py --variant f32 \
        --manifest-dir /tmp/gifteval_official --forecast-dir /tmp/gifteval_official/forecasts_f32
"""

import argparse
import csv
import json
import os
import struct
import warnings

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from gift_eval.data import Dataset  # noqa: E402
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss  # noqa: E402
from gluonts.model import evaluate_model  # noqa: E402
from gluonts.model.forecast import QuantileForecast  # noqa: E402

TRAINED_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
QUERY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def interpolate_quantiles(query_levels, orig_levels, orig_values):
    """Row-wise piecewise-linear interpolation, matching
    `t0.quantile.interpolate_quantiles` -- see `tools/score_gifteval.py`'s
    copy of this function for the verification note."""
    order = np.argsort(orig_levels)
    levels = orig_levels[order]
    values = orig_values[:, order]
    levels_padded = np.concatenate([[0.0], levels, [1.0]])
    values_padded = np.concatenate([values[:, :1], values, values[:, -1:]], axis=1)
    out = np.zeros((orig_values.shape[0], len(query_levels)), dtype=np.float32)
    for qi, q in enumerate(query_levels):
        upper = np.searchsorted(levels_padded, q, side="right")
        upper = min(max(upper, 1), len(levels_padded) - 1)
        lower = upper - 1
        lo_l, hi_l = levels_padded[lower], levels_padded[upper]
        w = 0.0 if hi_l == lo_l else (q - lo_l) / (hi_l - lo_l)
        out[:, qi] = values_padded[:, lower] + w * (values_padded[:, upper] - values_padded[:, lower])
    return out


def read_f32(path, n):
    with open(path, "rb") as f:
        data = f.read()
    return np.array(struct.unpack(f"<{n}f", data), dtype=np.float32)


class _FullTestData:
    def __init__(self, input_list, label_list):
        self.input = input_list
        self.label = label_list


def score_task(task, forecasts_9q):
    """forecasts_9q: [n_windows, horizon, 9], QUERY_LEVELS order, aligned
    with task["windows"] order (the FULL test split here, so indices == range(n))."""
    ds = Dataset(name=task["gift_eval_name"], term="short", to_univariate=False)
    if ds.target_dim != 1:
        ds = Dataset(name=task["gift_eval_name"], term="short", to_univariate=True)
    all_inputs = list(ds.test_data.input)
    all_labels = list(ds.test_data.label)
    indices = [w["index"] for w in task["windows"]]
    assert len(indices) == task["n_windows"] == forecasts_9q.shape[0]

    forecasts = []
    for out_i, idx in enumerate(indices):
        entry = all_inputs[idx]
        arr = forecasts_9q[out_i].T  # [9, horizon]
        target = np.atleast_1d(np.asarray(entry["target"], dtype=np.float32)).ravel()
        forecasts.append(
            QuantileForecast(
                forecast_arrays=arr,
                forecast_keys=[str(q) for q in QUERY_LEVELS],
                start_date=entry["start"] + len(target),
                item_id=str(idx),
            )
        )

    sub_test_data = _FullTestData([all_inputs[i] for i in indices], [all_labels[i] for i in indices])

    class FixedPredictor:
        prediction_length = task["horizon"]

        def predict(self, dataset):
            return iter(forecasts)

    res = evaluate_model(
        FixedPredictor(),
        test_data=sub_test_data,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=QUERY_LEVELS)],
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=task["seasonality"],
    )
    return float(res["MASE[0.5]"].iloc[0]), float(res["mean_weighted_sum_quantile_loss"].iloc[0])


def load_results_csv(path):
    """Read a gift-eval-style `all_results.csv` (our own, or theirs, e.g.
    `results/seasonal_naive/all_results.csv` from the gift-eval repo) and
    return {config: (mase, crps)} keyed by the `dataset` column verbatim
    (e.g. "loop_seattle/5T/short") -- this is also the GIFT-Eval leaderboard's
    join key, see `--normalize-by`'s docstring below."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["dataset"]] = (
                float(r["eval_metrics/MASE[0.5]"]),
                float(r["eval_metrics/mean_weighted_sum_quantile_loss"]),
            )
    return out


def geo_mean(values):
    a = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(a))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="label only, used in the printed summary line")
    ap.add_argument("--manifest-dir", help="forecast-recomputation mode: mutually exclusive with --results-csv")
    ap.add_argument("--forecast-dir", help="forecast-recomputation mode: mutually exclusive with --results-csv")
    ap.add_argument(
        "--results-csv",
        help="score an already-computed gift-eval-style all_results.csv directly "
        "(dataset, eval_metrics/MASE[0.5], eval_metrics/mean_weighted_sum_quantile_loss "
        "columns) instead of recomputing forecasts from --manifest-dir/--forecast-dir",
    )
    ap.add_argument(
        "--normalize-by",
        help="path to a seasonal-naive all_results.csv (e.g. gift-eval's "
        "results/seasonal_naive/all_results.csv) to additionally print the "
        "GIFT-Eval-leaderboard-comparable normalised aggregate: per config, "
        "divide this run's MASE/CRPS by the seasonal-naive baseline's MASE/CRPS "
        "on that same config (joined on the `dataset` column, e.g. "
        "\"loop_seattle/5T/short\"), then take the geometric mean "
        "(prod(x)**(1/n)) over configs. This is the exact normalization the "
        "GIFT-Eval leaderboard notebook applies before its geometric mean -- see "
        "SalesforceAIResearch/gift-eval notebooks/zeus.ipynb (commit "
        "9a014e9e8ea130ba39c100c60d5dcbab7db57ac9), the 'Print Results' cell: "
        "`df['normalized MASE'] = df['eval_metrics/MASE[0.5]'] / seasonal_naive_mase[idx]` "
        "(same for CRPS), then `geo_mean(df['normalized MASE'])` with "
        "`geo_mean = lambda a: np.array(a).prod()**(1.0/len(a))` -- algebraically "
        "identical to this script's own `exp(mean(log(x)))` raw aggregate, just "
        "applied to the normalised ratios instead of the raw values. Every one "
        "of the 97 GIFT-Eval configs must have a matching row in both CSVs, or "
        "this exits with an error listing the unmatched keys.",
    )
    args = ap.parse_args()

    if args.results_csv:
        results = load_results_csv(args.results_csv)
        rows = [(config, None, mase, crps) for config, (mase, crps) in results.items()]
    else:
        if not (args.manifest_dir and args.forecast_dir):
            ap.error("either --results-csv, or both --manifest-dir and --forecast-dir, are required")
        with open(os.path.join(args.manifest_dir, "manifest.json")) as f:
            manifest = json.load(f)

        rows = []
        for task in manifest["tasks"]:
            safe = task["config"].replace("/", "_")
            path = os.path.join(args.forecast_dir, f"{safe}.f32")
            n = task["n_windows"] * task["horizon"] * len(TRAINED_LEVELS)
            flat = read_f32(path, n).reshape(task["n_windows"], task["horizon"], len(TRAINED_LEVELS))
            flat2 = flat.reshape(-1, len(TRAINED_LEVELS))
            interp = interpolate_quantiles(QUERY_LEVELS, TRAINED_LEVELS, flat2)
            preds = interp.reshape(task["n_windows"], task["horizon"], len(QUERY_LEVELS))
            mase, crps = score_task(task, preds)
            print(f"{task['config']}: n={task['n_windows']} MASE={mase:.4f} CRPS={crps:.4f}")
            rows.append((task["config"], task["n_windows"], mase, crps))

    mases = np.array([r[2] for r in rows])
    crpses = np.array([r[3] for r in rows])
    agg_mase = geo_mean(mases)
    agg_crps = geo_mean(crpses)
    print(
        f"\n{args.variant} raw aggregate (geometric mean over {len(rows)} configs, "
        f"NOT normalized by seasonal-naive, not comparable to the GIFT-Eval "
        f"leaderboard/model-card numbers): MASE={agg_mase:.4f} CRPS={agg_crps:.4f}"
    )

    if args.normalize_by:
        naive = load_results_csv(args.normalize_by)
        unmatched = [config for config, *_ in rows if config not in naive]
        if unmatched:
            print(f"\nERROR: {len(unmatched)} config(s) missing from --normalize-by baseline:")
            for config in unmatched:
                print(f"  {config}")
            raise SystemExit(1)

        norm_mases = []
        norm_crpses = []
        print(f"\n{args.variant} per-config normalised (this run's MASE/CRPS / seasonal-naive's):")
        for config, _n, mase, crps in rows:
            naive_mase, naive_crps = naive[config]
            nm, nc = mase / naive_mase, crps / naive_crps
            norm_mases.append(nm)
            norm_crpses.append(nc)
            print(f"{config}: normalized_MASE={nm:.4f} normalized_CRPS={nc:.4f}")

        agg_norm_mase = geo_mean(norm_mases)
        agg_norm_crps = geo_mean(norm_crpses)
        print(
            f"\n{args.variant} normalised aggregate (geometric mean over {len(rows)} configs, "
            f"comparable to the GIFT-Eval leaderboard / model-card numbers): "
            f"MASE={agg_norm_mase:.4f} CRPS={agg_norm_crps:.4f}"
        )


if __name__ == "__main__":
    main()
