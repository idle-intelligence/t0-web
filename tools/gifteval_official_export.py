"""Export the SAME 8 GIFT-Eval configs as `tools/gifteval_subset.py`, but at
(close to) the officially published protocol: full test split (no
`MAX_WINDOWS_PER_TASK` subsampling) and `CONTEXT_LENGTH=8192` (trailing
window of the series, full history if shorter) instead of the 512-step cap.

This is the manifest `t0-cli gifteval --backend fast` (via `--manifest`)
consumes for the "official-protocol subset" row in `docs/BENCHMARKS.md` --
see `docs/runs/2026-09-20-rollout.md`'s rollout-fixed session for why this
needed `forecast_rollout`'s AR loop working at all: earlier attempts landed
in `docs/runs/2026-09-19-official-protocol-subset.md` (their own PyTorch
code, not ours) because our engines had no autoregressive rollout yet.

Deviation from CONTEXT_LENGTH=8192: `t0-fast`'s attention kernel
(`crates/t0-fast/src/shaders/attention.wgsl`) caps the jointly-attended
patch count at `MAX_SEQ=256` (WebGPU's workgroup-invocation limit). A
time-layer forward attends over `ceil(context/32) + ceil(horizon/32)`
patches; 8192 is exactly 256 patches, which plus even a 1-patch horizon
exceeds the cap. This script uses 8160 (255 patches) instead, which only
shortens one of the 8 configs (`saugeenday`, whose full history exceeds
8192 anyway) by 32 of its oldest samples out of >20000 -- not the other 7,
whose full history is well under 8160.

Usage:
    .venv/bin/python3 tools/gifteval_official_export.py --out-dir /tmp/gifteval_official
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
from gluonts.time_feature import get_seasonality  # noqa: E402

CONTEXT_LENGTH = 8160  # see module docstring

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
    windows = []
    for i, inp in enumerate(all_inputs):
        target = np.atleast_1d(np.asarray(inp["target"], dtype=np.float32)).ravel()
        ctx = target[-CONTEXT_LENGTH:] if len(target) > CONTEXT_LENGTH else target
        ctx = np.ascontiguousarray(ctx, dtype=np.float32)
        ctx_file = f"{ds_key}_{ds.freq}_{i}_context.f32"
        write_f32(os.path.join(out_dir, ctx_file), ctx.tolist())
        windows.append({"index": i, "context_len": len(ctx), "context_file": ctx_file})

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {"context_length": CONTEXT_LENGTH, "tasks": []}
    for name in CONFIGS:
        task = export_config(name, args.out_dir)
        print(f"{task['config']}: {task['n_windows']} windows, horizon={task['horizon']}, max_ctx={max(w['context_len'] for w in task['windows'])}")
        manifest["tasks"].append(task)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"wrote manifest + {sum(t['n_windows'] for t in manifest['tasks'])} windows to {args.out_dir}")


if __name__ == "__main__":
    main()
