"""Export a GIFT-Eval subset's test windows to little-endian f32 + a JSON
manifest that `t0-cli gifteval` reads, mirroring `fixtures/`'s style.

Python only: `gift_eval.data.Dataset` (the official loader, GluonTS-backed)
is itself a Python/Arrow class — re-deriving its train/test window split any
other way would be re-implementing the benchmark harness we're trying to
match, telling us nothing (same rationale as `make_fixtures.py`).

Deviation from the official protocol, documented here and in
docs/runs/2026-09-19-gifteval-subset.md: the reference notebook uses
CONTEXT_LENGTH=8192; our Burn port has no autoregressive rollout yet
(README "What's not done yet"), so a single forward pass is capped at
max_horizon=1024 total (context+horizon, patch-padded). We cap context to
CONTEXT_CAP=512 for BOTH the reference and our model on every window, so
the reference-vs-ours comparison stays apples-to-apples even though neither
side is run at the officially-published 8192-context setting.

Usage:
    .venv/bin/python3 tools/gifteval_subset.py
"""

import json
import os
import struct

import numpy as np
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.time_feature import get_seasonality

load_dotenv()

CONTEXT_CAP = 512  # see module docstring
# Full-test-set window counts for this subset run 20..1850 per task
# (4099 total). A single non-batched forward pass on the CPU/ndarray
# backend costs ~0.1-0.3s (see docs/runs/2026-09-19-gifteval-subset.md);
# 4099 windows x 4 weight variants (F32/f16/Q8_0/Q4_0) would run
# ~1.5-3.5h. We deterministically stride-subsample each task down to at
# most MAX_WINDOWS_PER_TASK windows (evenly spaced, not the first N, so
# short/long-history windows are both represented) to keep the whole
# subset x quant-ladder sweep under ~15 minutes. Documented deviation,
# same rationale/place as the context-length cap above.
MAX_WINDOWS_PER_TASK = 60
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "gifteval")

# (dataset/freq, domain) — see docs/runs/2026-09-19-gifteval-subset.md for
# why each was picked.
CONFIGS = [
    "electricity/D",
    "electricity/W",
    "solar/D",
    "solar/W",
    "jena_weather/D",
    "LOOP_SEATTLE/D",
    "saugeenday/D",
    "us_births/D",
]


def write_f32(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(arr)}f", *arr))


def export_config(name, out_dir):
    to_univariate = Dataset(name=name, term="short", to_univariate=False).target_dim != 1
    ds = Dataset(name=name, term="short", to_univariate=to_univariate)
    season = get_seasonality(ds.freq)
    horizon = ds.prediction_length
    ds_key = name.split("/")[0]
    config_name = f"{ds_key.lower()}/{ds.freq}/short"

    all_inputs = list(ds.test_data.input)
    all_labels = list(ds.test_data.label)
    n_total = len(all_inputs)
    if n_total > MAX_WINDOWS_PER_TASK:
        idx = np.linspace(0, n_total - 1, MAX_WINDOWS_PER_TASK).round().astype(int)
        idx = sorted(set(idx.tolist()))
    else:
        idx = list(range(n_total))

    windows = []
    for i in idx:
        inp, label = all_inputs[i], all_labels[i]
        target = np.asarray(inp["target"], dtype=np.float32)
        target = np.atleast_1d(target)
        if target.ndim > 1:
            target = target[0]  # to_univariate splits multivariate already; defensive
        ctx = target[-CONTEXT_CAP:] if len(target) > CONTEXT_CAP else target
        ctx = np.ascontiguousarray(ctx, dtype=np.float32)
        future = np.atleast_1d(np.asarray(label["target"], dtype=np.float32))
        if future.ndim > 1:
            future = future[0]
        assert len(future) == horizon, f"{name} window {i}: label len {len(future)} != horizon {horizon}"

        item_id = inp.get("item_id", str(i))
        ctx_file = f"{ds_key}_{ds.freq}_{i}_context.f32"
        fut_file = f"{ds_key}_{ds.freq}_{i}_future.f32"
        write_f32(os.path.join(out_dir, ctx_file), ctx.tolist())
        write_f32(os.path.join(out_dir, fut_file), future.tolist())
        windows.append(
            {
                "index": i,  # position within the FULL (unsampled) ds.test_data
                "item_id": str(item_id),
                "context_len": len(ctx),
                "context_file": ctx_file,
                "future_file": fut_file,
            }
        )

    return {
        "config": config_name,
        "gift_eval_name": name,
        "freq": ds.freq,
        "horizon": horizon,
        "seasonality": season,
        "n_windows": len(windows),
        "windows": windows,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "context_cap": CONTEXT_CAP,
        "quantile_levels_trained": [0.1, 0.25, 0.5, 0.75, 0.9],
        "quantile_levels_gifteval": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "tasks": [],
    }
    for name in CONFIGS:
        task = export_config(name, OUT_DIR)
        print(f"{task['config']}: {task['n_windows']} windows, horizon={task['horizon']}, seasonality={task['seasonality']}")
        manifest["tasks"].append(task)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest + {sum(t['n_windows'] for t in manifest['tasks'])} windows to {OUT_DIR}")


if __name__ == "__main__":
    main()
