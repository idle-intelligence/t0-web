"""Same official-protocol GIFT-Eval subset as
`.claude/worktrees/fullsuite/tools/fullsuite/run_gifteval.py`, adapted for
t0-beta. That script is hardcoded to t0-alpha (T0_ALPHA_DIR, ONNX_PATH,
`t0-alpha-{variant}.gguf` naming) and lives in a different worktree that
this task must not edit (repo rule: never edit another worktree from this
one), so this is a beta-scoped copy rather than a `--model` flag added
in place. Same "their code, their protocol": `gift_eval.data.Dataset`
windowing, `gluonts.model.evaluate_model` + their metrics,
`t0.evaluation.T0Predictor` verbatim, full CONTEXT_LENGTH=8192, no window
subsampling. Only the weights/variant plumbing differs (t0-beta dir,
21-level quantiles per its config, beta gguf files).

`dequant_gguf.read_gguf` is imported read-only from the fullsuite worktree
(generic, no alpha-specific hardcoding) rather than duplicated.

venv: this repo's .venv (reused read-only, same
as run_gifteval.py). Launch:

    .venv/bin/python3 \\
        tools/fullsuite_gifteval_beta.py --variant f32 --max-minutes 20
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
from dotenv import load_dotenv

FULLSUITE_TOOLS = os.environ.get(
    "FULLSUITE_TOOLS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "fullsuite"),
)
sys.path.insert(0, FULLSUITE_TOOLS)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0_BETA_DIR = os.environ.get(
    "T0_BETA_DIR",
    os.path.expanduser("~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-beta"),
)
GGUF_DIR = os.path.join(ROOT, "fixtures", "beta", "gguf")

CONTEXT_LENGTH = 8192  # same as run_gifteval.py, their notebook's own value
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH_SIZE = 64

CONFIGS = [
    "us_births/D",
    "saugeenday/D",
    "jena_weather/D",
    "solar/W",
    "solar/D",
    "LOOP_SEATTLE/D",
    "electricity/W",
    "electricity/D",
]

CSV_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]


def load_pretty_names():
    return {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }


def build_predictor(variant, prediction_length, device="cpu"):
    from t0 import T0Forecaster
    from t0.evaluation import T0Predictor

    model = T0Forecaster.from_pretrained(T0_BETA_DIR).to(device).eval()
    if variant != "f32":
        from dequant_gguf import read_gguf

        gguf_path = os.path.join(GGUF_DIR, f"t0-beta-{variant}.gguf")
        _, tensors = read_gguf(gguf_path)
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
    return T0Predictor(
        model,
        prediction_length=prediction_length,
        quantile_levels=QUANTILE_LEVELS,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        show_progress=True,
    ), model


def run_config(variant, ds_name, pretty_names, term="short"):
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]

    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/")
        ds_key = pretty_names.get(ds_key.lower(), ds_key.lower())
    else:
        ds_key = pretty_names.get(ds_name.lower(), ds_name.lower())
        dataset_properties_map_tmp = json.load(open(os.path.join("/private/tmp/gift-eval/notebooks", "dataset_properties.json")))
        ds_freq = dataset_properties_map_tmp[ds_key]["frequency"]
    ds_config = f"{ds_key}/{ds_freq}/{term}"

    dataset_properties_map = json.load(open(os.path.join("/private/tmp/gift-eval/notebooks", "dataset_properties.json")))

    to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    season_length = get_seasonality(dataset.freq)
    n_windows = len(dataset.test_data)
    print(f"{ds_config}: {n_windows} windows, horizon={dataset.prediction_length}, seasonality={season_length}")

    predictor, _ = build_predictor(variant_ctx["variant"], dataset.prediction_length)

    t0 = time.monotonic()
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    elapsed = time.monotonic() - t0

    row = {
        "dataset": ds_config,
        "model": variant_ctx["variant"],
        "eval_metrics/MSE[mean]": res["MSE[mean]"].iloc[0],
        "eval_metrics/MSE[0.5]": res["MSE[0.5]"].iloc[0],
        "eval_metrics/MAE[0.5]": res["MAE[0.5]"].iloc[0],
        "eval_metrics/MASE[0.5]": res["MASE[0.5]"].iloc[0],
        "eval_metrics/MAPE[0.5]": res["MAPE[0.5]"].iloc[0],
        "eval_metrics/sMAPE[0.5]": res["sMAPE[0.5]"].iloc[0],
        "eval_metrics/MSIS": res["MSIS"].iloc[0],
        "eval_metrics/RMSE[mean]": res["RMSE[mean]"].iloc[0],
        "eval_metrics/NRMSE[mean]": res["NRMSE[mean]"].iloc[0],
        "eval_metrics/ND[0.5]": res["ND[0.5]"].iloc[0],
        "eval_metrics/mean_weighted_sum_quantile_loss": res["mean_weighted_sum_quantile_loss"].iloc[0],
        "domain": dataset_properties_map[ds_key]["domain"],
        "num_variates": dataset_properties_map[ds_key]["num_variates"],
    }
    return row, n_windows, elapsed


variant_ctx = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["f32", "q8_0", "q4_0"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    args = ap.parse_args()

    variant_ctx["variant"] = args.variant
    pretty_names = load_pretty_names()

    out_dir = os.path.join(ROOT, "results", "beta", args.variant)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "all_results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    specs = [(c, "short") for c in (args.configs or CONFIGS)]

    budget_s = args.max_minutes * 60.0
    wall_start = time.monotonic()
    timings = []
    stopped_early = False

    for ds_name, term in specs:
        if time.monotonic() - wall_start > budget_s:
            print(f"budget of {args.max_minutes} min exceeded, stopping before {ds_name}/{term}")
            stopped_early = True
            break
        row, n_windows, elapsed = run_config(args.variant, ds_name, pretty_names, term=term)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        per_window = elapsed / n_windows
        timings.append({"config": row["dataset"], "n_windows": n_windows, "elapsed_s": elapsed, "s_per_window": per_window})
        print(
            f"{row['dataset']}: MASE={row['eval_metrics/MASE[0.5]']:.4f} "
            f"CRPS={row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} "
            f"({elapsed:.1f}s, {n_windows} windows, {per_window:.3f}s/window)"
        )

    summary = {
        "variant": args.variant,
        "context_length": CONTEXT_LENGTH,
        "max_minutes": args.max_minutes,
        "stopped_early": stopped_early,
        "timings": timings,
    }
    summary_path = os.path.join(out_dir, "timing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
