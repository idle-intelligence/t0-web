#!/usr/bin/env python3
"""Generate parity fixtures against the reference PyTorch t0-alpha model.

Python is used here only because the reference implementation
(`theforecastingcompany/tfc-t0`, PyPI package `t0`) is itself PyTorch —
re-deriving "golden" outputs any other way (e.g. hand-porting the model to
NumPy) would just be re-implementing the thing we're trying to check, so it
would tell us nothing. This script never touches the Rust port; it only
calls the installed reference package and dumps its inputs/outputs.

Requires: `python3 -m venv .venv && . .venv/bin/activate && pip install tfc-t0`
(the PyPI package name is `tfc-t0`; the importable module is `t0`).

Usage: `python3 tools/make_fixtures.py`
"""

import json
from pathlib import Path

import numpy as np
import torch

from t0.model.model import T0Forecaster

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
FIXTURES.mkdir(exist_ok=True)

CONTEXT_LEN = 512
HORIZON = 96
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)  # == the trained levels, so no
# rollout interpolation/extrapolation happens (see docs/reports/t0-alpha.md §5).


def write_f32(path: Path, arr: np.ndarray) -> None:
    data = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(data.tobytes())


def make_case(
    name: str,
    context: np.ndarray,
    model: T0Forecaster,
    *,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> dict:
    """context: (v, context_len) float32, NaN = missing observation."""
    v = context.shape[0]
    ctx_tensor = torch.from_numpy(context.astype(np.float32))
    if v > 1:
        ctx_tensor = ctx_tensor.unsqueeze(0)  # (V, T) -> (1, V, T), one joint multivariate sample

    captured = {}

    def hook(name):
        def _fn(_module, _inputs, output):
            captured[name] = output.detach().to(torch.float32).cpu().numpy()

        return _fn

    h1 = model.patch_encoder.register_forward_hook(hook("patch_embedding"))
    h2 = model.transformer.layers[0].register_forward_hook(hook("layer0_output"))
    try:
        forecast = model.predict(
            ctx_tensor,
            horizon=horizon,
            quantile_levels=quantile_levels,
            context_length=context_len,
        )
    finally:
        h1.remove()
        h2.remove()

    quantiles = forecast.quantiles.detach().to(torch.float32).cpu().numpy()
    # forecast.quantiles is (targets, horizon, Q) for the (1,T) case or
    # (1, V, horizon, Q) for the (1,V,T) case; normalize to (v, horizon, Q).
    quantiles = quantiles.reshape(v, horizon, len(quantile_levels))

    patch_embedding = captured["patch_embedding"]  # (v, n_patches, embed_dim)
    layer0_output = captured["layer0_output"]
    n_patches = patch_embedding.shape[1]

    write_f32(FIXTURES / f"{name}_context.f32", context)
    write_f32(FIXTURES / f"{name}_quantiles.f32", quantiles)
    write_f32(FIXTURES / f"{name}_patch_embedding.f32", patch_embedding)
    write_f32(FIXTURES / f"{name}_layer0_output.f32", layer0_output)

    return {
        "name": name,
        "v": v,
        "n_patches": int(n_patches),
        "context_len": context_len,
        "horizon": horizon,
        "context_file": f"{name}_context.f32",
        "quantiles_file": f"{name}_quantiles.f32",
        "patch_embedding_file": f"{name}_patch_embedding.f32",
        "layer0_output_file": f"{name}_layer0_output.f32",
    }


def main() -> None:
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha")
    model.eval()

    rng = np.random.default_rng(0)
    t = np.arange(CONTEXT_LEN, dtype=np.float32)

    cases = []

    # Case 0: univariate sine + noise.
    ctx0 = (np.sin(2 * np.pi * t / 48.0) + 0.1 * rng.standard_normal(CONTEXT_LEN)).astype(np.float32)
    cases.append(make_case("case0_univariate_sine", ctx0[None, :], model))

    # Case 1: 3-variate, different frequencies, forecast jointly.
    ctx1 = np.stack(
        [
            np.sin(2 * np.pi * t / 24.0),
            np.sin(2 * np.pi * t / 48.0 + 0.7),
            np.sin(2 * np.pi * t / 96.0 + 1.3) * 0.5,
        ]
    ).astype(np.float32)
    ctx1 += 0.05 * rng.standard_normal(ctx1.shape).astype(np.float32)
    cases.append(make_case("case1_trivariate_freqs", ctx1, model))

    # Case 2: univariate with a NaN gap in the middle of the context.
    ctx2 = (np.sin(2 * np.pi * t / 60.0) + 0.1 * rng.standard_normal(CONTEXT_LEN)).astype(np.float32)
    ctx2[200:220] = np.nan
    cases.append(make_case("case2_masked_gap", ctx2[None, :], model))

    manifest = {
        "reference": "tfc-t0 (PyPI) t0.model.model.T0Forecaster, theforecastingcompany/t0-alpha",
        "context_len": CONTEXT_LEN,
        "horizon": HORIZON,
        "patch_size": model.patch_size,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cases": cases,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(cases)} fixtures to {FIXTURES}")

    # Long-horizon case: context 4096, well past model.max_horizon (1024) so
    # T0Forecaster.predict actually drives RolloutManager's autoregressive
    # loop (a horizon <= max_horizon, e.g. 480 as originally asked for,
    # returns from RolloutManager.predict's first block and never touches
    # the AR path at all -- see model/rollout.py's `if prediction_length <=
    # horizon: return` -- so this exercises the rollout with horizon=1200
    # instead, which needs two AR blocks after the max_horizon=1024 first
    # one: round_up(1200,32)=1200 > 1024, so block 0 covers 1024, then one
    # more block covers the remaining 176).
    LONG_CONTEXT_LEN = 4096
    LONG_HORIZON = 1200
    t_long = np.arange(LONG_CONTEXT_LEN, dtype=np.float32)
    ctx_long = (np.sin(2 * np.pi * t_long / 288.0) + 0.1 * rng.standard_normal(LONG_CONTEXT_LEN)).astype(np.float32)
    long_case = make_case(
        "case_long_rollout",
        ctx_long[None, :],
        model,
        context_len=LONG_CONTEXT_LEN,
        horizon=LONG_HORIZON,
        quantile_levels=QUANTILE_LEVELS,
    )
    long_manifest = {
        "reference": "tfc-t0 (PyPI) t0.model.model.T0Forecaster, theforecastingcompany/t0-alpha, "
        "RolloutManager autoregressive path (t0/model/rollout.py)",
        "patch_size": model.patch_size,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cases": [long_case],
    }
    (FIXTURES / "manifest_long.json").write_text(json.dumps(long_manifest, indent=2))
    print(f"wrote long-horizon fixture ({long_case['context_len']=}, {long_case['horizon']=}) to {FIXTURES}")


if __name__ == "__main__":
    main()
