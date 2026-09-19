# t0-web

Port of The Forecasting Company's [t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha) time-series forecaster to Burn/wgpu, native and WASM+WebGPU. See `GOAL.md` for the project plan, `CLAUDE.md` for working rules, `docs/reports/t0-alpha.md` for the architecture research.

## Status: Milestone 0 (native numerical parity, F32)

`crates/t0-core` is a backend-generic Burn 0.20 port of `T0Forecaster` (see [`tfc-t0` PyPI package](https://pypi.org/project/tfc-t0/), the official PyTorch implementation) matching the reference to 1.2e-6 max-abs error on 3 fixtures (`docs/BENCHMARKS.md`). No quantization, no WGSL kernels, no browser page yet — that's Milestone 1+.

## Layout

- `crates/t0-core` — the model, generic over `burn::tensor::backend::Backend`. Safetensors loader, RMSNorm, RoPE (interleaved-pair/xpos, matching `rotary_embedding_torch`), QK-norm multi-head attention (time + group/cross-variate), SwiGLU MLP, causal per-timestep scaler, patch encoder, quantile head. No backend selection here — that's the binary's job.
- `crates/cli` — `t0-cli`, native binary. `ndarray` feature (default) or `wgpu` feature picks the backend. Commands: `forecast --fixture N`, `parity`.
- `tools/make_fixtures.py` — generates `fixtures/` from the reference PyTorch model. Python only because the reference is PyTorch (see its docstring).
- `fixtures/` — committed (small: ~430 KB), one manifest + per-case context/output/intermediate-activation binaries (little-endian f32).
- `docs/reports/t0-alpha.md` — architecture research report.
- `docs/BENCHMARKS.md`, `docs/runs/` — research log.

## Build

Native (default `ndarray` backend):
```
cargo build --release
cargo test
```

Native with `wgpu` backend (feature-gated in the `cli` crate, not the library):
```
cargo build -p t0-cli --release --no-default-features --features wgpu
```

WASM (compile check only in this milestone — no browser page yet):
```
cargo build -p t0-core --target wasm32-unknown-unknown
```
This needs `.cargo/config.toml`'s `--cfg getrandom_backend="wasm_js"` rustflag (already committed) — Burn's dependency chain pulls in `rand`/`getrandom` even though this inference-only crate never calls it.

## Fixtures

`tools/make_fixtures.py` requires the reference PyTorch package:
```
python3 -m venv .venv && . .venv/bin/activate && pip install tfc-t0
python3 tools/make_fixtures.py
```
Python is used here only because the reference implementation is itself PyTorch — re-deriving golden outputs any other way would just be re-implementing the thing being checked, telling us nothing. This never runs as part of the Rust build; fixtures are committed.

## Parity gate

Needs the t0-alpha checkpoint (not committed — weights are never committed, per `CLAUDE.md`):
```
hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha
cargo run -p t0-cli --release -- parity \
  --fixtures fixtures \
  --weights ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/model.safetensors \
  --config ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/config.json
```
Gate: quantile output within 1e-4 max-abs of the reference on every fixture. Current result: 1.2e-6 (`docs/BENCHMARKS.md`).

The same checkpoint is needed for `cargo test -- --ignored` (`crates/t0-core/tests/{loader,parity}.rs`).

## What's not done yet

- `FUTURE` covariate rows in the scaler, autoregressive rollout past 1024 steps, quantile interpolation/extrapolation for non-trained levels — see `docs/runs/2026-09-19-parity.md`'s last section.
- Q8/Q4 quantization and the WGSL kernels.
- The wgpu/WASM *runtime* path (only compiles today; no browser execution, no `web/` page).
- Reused nothing verbatim from `llm-web`: its `crates/llm-wasm/src/model.rs` is hardcoded to the `Wgpu` backend and the Qwen2 rotate-half RoPE convention (different from t0-alpha's interleaved-pair xpos RoPE), so a fresh backend-generic implementation was more direct than adapting it. The `getrandom/wasm_js` + `.cargo/config.toml` cfg-flag fix for the wasm32 build *was* copied from `llm-wasm/Cargo.toml`'s `web` feature.
