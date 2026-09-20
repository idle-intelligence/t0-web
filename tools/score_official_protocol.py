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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="label only, used in the printed summary line")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--forecast-dir", required=True)
    args = ap.parse_args()

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
    agg_mase = float(np.exp(np.mean(np.log(mases))))
    agg_crps = float(np.exp(np.mean(np.log(crpses))))
    print(f"\n{args.variant} aggregate (geometric mean over {len(rows)} tasks): MASE={agg_mase:.4f} CRPS={agg_crps:.4f}")


if __name__ == "__main__":
    main()
