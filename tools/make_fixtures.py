#!/usr/bin/env python3
"""Generate parity fixtures against the reference PyTorch t0-alpha/t0-beta model.

Python is used here only because the reference implementation
(`theforecastingcompany/tfc-t0`, PyPI package `t0`) is itself PyTorch —
re-deriving "golden" outputs any other way (e.g. hand-porting the model to
NumPy) would just be re-implementing the thing we're trying to check, so it
would tell us nothing. This script never touches the Rust port; it only
calls the installed reference package and dumps its inputs/outputs.

Requires: `python3 -m venv .venv && . .venv/bin/activate && pip install tfc-t0`
(the PyPI package name is `tfc-t0`; the importable module is `t0`).

Usage: `python3 tools/make_fixtures.py [alpha|beta]` (default: alpha).
alpha writes to `fixtures/`, beta to `fixtures/beta/` (native quantile
levels differ: 5 for alpha, 21 for beta — using the model's own trained
levels avoids rollout interpolation/extrapolation, see
docs/reports/t0-alpha.md §5).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from t0.model.model import T0Forecaster

ROOT = Path(__file__).resolve().parent.parent

MODEL = sys.argv[1] if len(sys.argv) > 1 else "alpha"
assert MODEL in ("alpha", "beta"), f"unknown model: {MODEL}"

HF_REPO = f"theforecastingcompany/t0-{MODEL}"
FIXTURES = ROOT / "fixtures" if MODEL == "alpha" else ROOT / "fixtures" / "beta"
FIXTURES.mkdir(parents=True, exist_ok=True)

CONTEXT_LEN = 512
HORIZON = 96
QUANTILE_LEVELS = (
    (0.1, 0.25, 0.5, 0.75, 0.9)
    if MODEL == "alpha"
    else (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
)


def write_f32(path: Path, arr: np.ndarray) -> None:
    data = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(data.tobytes())


def make_case(name: str, context: np.ndarray, model: T0Forecaster) -> dict:
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
            horizon=HORIZON,
            quantile_levels=QUANTILE_LEVELS,
            context_length=CONTEXT_LEN,
        )
    finally:
        h1.remove()
        h2.remove()

    quantiles = forecast.quantiles.detach().to(torch.float32).cpu().numpy()
    # forecast.quantiles is (targets, horizon, Q) for the (1,T) case or
    # (1, V, horizon, Q) for the (1,V,T) case; normalize to (v, horizon, Q).
    quantiles = quantiles.reshape(v, HORIZON, len(QUANTILE_LEVELS))

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
        "context_file": f"{name}_context.f32",
        "quantiles_file": f"{name}_quantiles.f32",
        "patch_embedding_file": f"{name}_patch_embedding.f32",
        "layer0_output_file": f"{name}_layer0_output.f32",
    }


def main() -> None:
    model = T0Forecaster.from_pretrained(HF_REPO)
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
        "reference": f"tfc-t0 (PyPI) t0.model.model.T0Forecaster, {HF_REPO}",
        "context_len": CONTEXT_LEN,
        "horizon": HORIZON,
        "patch_size": model.patch_size,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cases": cases,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(cases)} fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
