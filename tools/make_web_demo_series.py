#!/usr/bin/env python3
"""Generates the one bundled fixture series for `web/`'s one-signal
instrument (`web/data/series.f32`).

Not a real dataset (avoids the "weights/datasets never committed" question
entirely) and not one of `fixtures/*_context.f32` either,
because those stop at t=512 with no true continuation to compare a forecast
against. This is a deterministic synthetic daily-like series (no RNG, so
it's exactly reproducible from this script alone): a slow seasonal sine, a
faster secondary sine, a mild trend, and a small high-frequency wiggle
standing in for noise — same spirit as `crates/cli/src/main.rs`'s
`synthetic_case` (drift benchmark), long enough that the forecast-origin
slider has real room to move and a real future to check the fan against.

Usage: `python3 tools/make_web_demo_series.py`
"""
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data" / "series.f32"

N = 640  # total points; slider moves the forecast origin within this


def value(t: float) -> float:
    return (
        3.0 * math.sin(2 * math.pi * t / 48.0)
        + 1.2 * math.sin(2 * math.pi * t / 17.0 + 0.3)
        + 0.01 * t
        + 0.15 * math.sin(t / 3.1)
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    series = [value(t) for t in range(N)]
    with open(OUT, "wb") as f:
        f.write(struct.pack(f"<{N}f", *series))
    print(f"wrote {OUT} ({N} points, {N * 4} bytes)")


if __name__ == "__main__":
    main()
