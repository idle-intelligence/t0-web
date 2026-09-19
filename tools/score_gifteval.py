"""Score our Rust forecasts (`t0-cli gifteval`'s output) and the PyTorch
reference (`t0-alpha` via `tfc-t0`) with the SAME gluonts metric code
(`gluonts.model.evaluate_model` + `MASE`/`MeanWeightedSumQuantileLoss`), on
the SAME GIFT-Eval windows (`fixtures/gifteval/manifest.json`,
`tools/gifteval_subset.py`'s output).

This is the metric half of docs/runs/2026-09-19-gifteval-subset.md's plan:
- reference: run tfc-t0's T0Forecaster.predict() directly per window (same
  windows, same 512-step context cap, same interpolation to the 9 GIFT-Eval
  query levels the reference model does internally).
- ours: read the .f32 files `t0-cli gifteval` wrote (5 trained quantile
  levels per window) and linearly interpolate to the same 9 query levels
  with the exact algorithm t0's own `t0.quantile.interpolate_quantiles`
  uses (piecewise-linear between sorted (level, value) pairs, edge-padded
  at 0/1) — reimplemented here in numpy since it's ~15 lines and avoids a
  torch dependency in the scoring path; verified against `t0.quantile` on
  one window below (`--check-interp`).

Usage:
    .venv/bin/python3 tools/score_gifteval.py --which reference
    .venv/bin/python3 tools/score_gifteval.py --which f32 --forecast-dir fixtures/gifteval/forecasts_f32
    .venv/bin/python3 tools/score_gifteval.py --which f16 --forecast-dir fixtures/gifteval/forecasts_f16
    .venv/bin/python3 tools/score_gifteval.py --which q8_0 --forecast-dir fixtures/gifteval/forecasts_q8_0
    .venv/bin/python3 tools/score_gifteval.py --which q4_0 --forecast-dir fixtures/gifteval/forecasts_q4_0
"""

import argparse
import json
import os
import struct
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gift_eval.data import Dataset

load_dotenv()

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "gifteval")
TRAINED_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
QUERY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONTEXT_CAP = 512


def interpolate_quantiles(query_levels, orig_levels, orig_values):
    """Row-wise piecewise-linear interpolation, matching
    `t0.quantile.interpolate_quantiles` (edge-padded at 0.0/1.0 with the
    boundary value, `searchsorted(right=True)` bracketing). `orig_values`:
    [n, len(orig_levels)]. Returns [n, len(query_levels)]."""
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


def load_manifest():
    with open(os.path.join(MANIFEST_DIR, "manifest.json")) as f:
        return json.load(f)


class _SubsampledTestData:
    """`ds.test_data` restricted to `task["windows"][*]["index"]` — GIFT-Eval
    subsamples the full test set for runtime (see
    docs/runs/2026-09-19-gifteval-subset.md's MAX_WINDOWS_PER_TASK note);
    `evaluate_forecasts` only needs `.input`/`.label` to be iterables of
    dicts with a numpy `"target"`, so plain lists suffice — no GluonTS
    dataset wrapping needed."""

    def __init__(self, input_list, label_list):
        self.input = input_list
        self.label = label_list


def score_task(task, forecasts_9q, seasonality):
    """forecasts_9q: [n_windows, horizon, 9] numpy array, in QUERY_LEVELS order,
    aligned with `task["windows"]` (already subsampled) order."""
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

    sub_test_data = _SubsampledTestData([all_inputs[i] for i in indices], [all_labels[i] for i in indices])

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
        seasonality=seasonality,
    )
    return float(res["MASE[0.5]"].iloc[0]), float(res["mean_weighted_sum_quantile_loss"].iloc[0])


def run_reference(manifest):
    from t0 import T0Forecaster

    model = T0Forecaster.from_pretrained(
        "~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha"
    ).to("cpu").eval()

    rows = []
    all_forecasts = {}
    for task in manifest["tasks"]:
        t0 = time.monotonic()
        preds = np.zeros((task["n_windows"], task["horizon"], len(QUERY_LEVELS)), dtype=np.float32)
        for i, w in enumerate(task["windows"]):
            ctx = read_f32(os.path.join(MANIFEST_DIR, w["context_file"]), w["context_len"])
            fc = model.predict(ctx, horizon=task["horizon"], quantile_levels=QUERY_LEVELS)
            preds[i] = fc.quantiles[0].numpy().astype(np.float32)  # [1, horizon, 9] -> [horizon, 9]
        elapsed = time.monotonic() - t0
        mase, crps = score_task(task, preds, task["seasonality"])
        print(f"{task['config']}: MASE={mase:.4f} CRPS={crps:.4f} ({elapsed:.1f}s, {task['n_windows']} windows)")
        rows.append({"config": task["config"], "n_windows": task["n_windows"], "MASE": mase, "CRPS": crps})
        all_forecasts[task["config"]] = preds
    return pd.DataFrame(rows), all_forecasts


def run_ours(manifest, forecast_dir):
    rows = []
    for task in manifest["tasks"]:
        safe_name = task["config"].replace("/", "_")
        path = os.path.join(forecast_dir, f"{safe_name}.f32")
        n = task["n_windows"] * task["horizon"] * len(TRAINED_LEVELS)
        flat = read_f32(path, n).reshape(task["n_windows"], task["horizon"], len(TRAINED_LEVELS))
        # interpolate each window's [horizon, 5] to [horizon, 9]
        flat2 = flat.reshape(-1, len(TRAINED_LEVELS))
        interp = interpolate_quantiles(QUERY_LEVELS, TRAINED_LEVELS, flat2)
        preds = interp.reshape(task["n_windows"], task["horizon"], len(QUERY_LEVELS))
        mase, crps = score_task(task, preds, task["seasonality"])
        print(f"{task['config']}: MASE={mase:.4f} CRPS={crps:.4f}")
        rows.append({"config": task["config"], "n_windows": task["n_windows"], "MASE": mase, "CRPS": crps})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["reference", "f32", "f16", "q8_0", "q4_0"])
    ap.add_argument("--forecast-dir", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    manifest = load_manifest()
    if args.which == "reference":
        df, _ = run_reference(manifest)
    else:
        df = run_ours(manifest, args.forecast_dir)

    agg_mase = float(np.exp(np.mean(np.log(df["MASE"]))))
    agg_crps = float(np.exp(np.mean(np.log(df["CRPS"]))))
    print(f"\n{args.which} aggregate (geometric mean over {len(df)} tasks): MASE={agg_mase:.4f} CRPS={agg_crps:.4f}")

    out_csv = args.out_csv or os.path.join(MANIFEST_DIR, f"scores_{args.which}.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
