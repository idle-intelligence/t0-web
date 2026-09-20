# t0-web

t0-alpha and t0-beta, The Forecasting Company's time-series foundation models, running in the browser: Rust compiled to WebAssembly, WebGPU inference with a CPU fallback, weights quantized to Q8_0 or Q4_0 GGUF.

[**Try the demo →**](https://idle-intelligence.github.io/t0-web/web/) · [**Compare page**](bench/compare/): the same engine next to the official ONNX INT8 export, runs locally (see `docs/runs/2026-09-20-compare-page.md`)

> **Disclaimer:** independent port by ilnmtlbnm@idle-intelligence, not affiliated with or endorsed by The Forecasting Company. Weights are quantized from [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha) and [theforecastingcompany/t0-beta](https://huggingface.co/theforecastingcompany/t0-beta) (Apache-2.0); forecasts may differ slightly from the PyTorch implementation.

## Status

- F32 port matches the PyTorch reference to 1.2e-6 max-abs error (alpha) and 3.2e-6 (beta), native and browser.
- Full GIFT-Eval protocol, 97 configs, normalized to Seasonal Naive: Q4_0 scores MASE 0.7334 / CRPS 0.4973, against our F32 control at 0.7255 / 0.4942 and the published t0-alpha card at 0.7240 / 0.4941 (`docs/runs/2026-09-20-official-protocol-full97-normalized.md`).
- Quantized weights on Hugging Face: [alpha Q4_0](https://huggingface.co/idle-intelligence/t0-alpha-q4_0-webgpu) · [alpha Q8_0](https://huggingface.co/idle-intelligence/t0-alpha-q8_0-webgpu) · [beta Q4_0](https://huggingface.co/idle-intelligence/t0-beta-q4_0-webgpu) · [beta Q8_0](https://huggingface.co/idle-intelligence/t0-beta-q8_0-webgpu).
- Architecture research in `docs/reports/t0-alpha.md` and `docs/reports/t0-alpha-anatomy.md`.

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

Needs the t0-alpha checkpoint (not committed; weights are never committed):
```
hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha
cargo run -p t0-cli --release -- parity \
  --fixtures fixtures \
  --weights ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/model.safetensors \
  --config ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha/config.json
```
Gate: quantile output within 1e-4 max-abs of the reference on every fixture. Current result: 1.2e-6 (`docs/BENCHMARKS.md`).

The same checkpoint is needed for `cargo test -- --ignored` (`crates/t0-core/tests/{loader,parity}.rs`).

## Status: Milestone 1a (quantized weights, native wgpu runtime, latency table)

### GGUF quantization

`t0-cli export-gguf` writes a GGUF v3 container (`crates/t0-core/src/gguf.rs`): llama.cpp's own block layouts — Q8_0 (32 values, f16 scale + 32 i8) and Q4_0 (32 values, f16 scale + 16 bytes of nibbles, matching `ggml_quantize_row_q4_0`'s exact rounding/sign convention) — plus the model config as `t0.*` GGUF metadata keys, so a GGUF file is self-describing (no separate `config.json` needed to load one).

**Which tensors are quantized** (`Weights::is_quantizable` in `crates/t0-core/src/weights.rs`): the four big per-layer matmuls — `attention.wQKV.weight`, `attention.wO.weight`, `mlp.0.weight`, `mlp.2.weight` — to the requested Q8_0/Q4_0/f16 scheme. Everything else stays f16: patch encoder, type embeddings, decoder, every norm (`q_norm`/`k_norm`/`attention_block.norm`/layer `norm`/`out_norm`), and `head.quantile_levels`. Rationale (`docs/reports/t0-alpha.md` §5): the patch encoder gates every downstream layer's accuracy, the decoder produces the quantile values directly, and norms/`quantile_levels` are tiny — quantizing them buys negligible size for real precision risk.

```
t0-cli export-gguf --weights <model.safetensors> --config <config.json> --quant f16|q8_0|q4_0 --out <file.gguf>
```

At load, every GGUF tensor is dequantized to f32 (`Weights::load_gguf`) — Burn 0.20.1's `burn-wgpu`/`cubecl-wgpu` 0.9.0 only implements `FloatElement` for `f32` (checked directly in `cubecl-wgpu-0.9.0/src/element.rs`), so there's no lower-precision compute path to dequantize into yet on either backend. Quantization here shrinks the file and load time, not the matmul dtype; the accuracy tradeoff (does Q4_0 hurt the actual forecast) is deferred to the GIFT-Eval step. Round-trip error against the PyTorch reference and the full latency table are in `docs/BENCHMARKS.md`.

### wgpu runtime

`t0-cli forecast|parity|bench --backend wgpu` (built with `cargo build -p t0-cli --release --no-default-features --features wgpu`) now actually executes on Metal via `burn-wgpu`, not just compiles. `--backend` is checked against the binary's compiled Cargo feature and errors out on a mismatch — backend selection stays a compile-time Cargo feature, not a runtime switch (no dynamic dispatch between two different `Backend` types in one binary). Confirmed: wgpu F32 parity matches the PyTorch reference to the same order of magnitude as ndarray (worst-case 1.6e-6 vs ndarray's 1.2e-6), and wgpu vs ndarray agree to within ~1e-4 on both F32 and f16-GGUF inputs.

### Batch path

`forecast_batch`/`TimeSeries::from_context_batch` run N independent signals in one forward pass by giving each its own group id — the existing cross-variate group-attention mask (group-id equality) isolates them, so no new attention code was needed. 8 batched copies of fixture 0 match 8x the single-signal forecast to <=1e-5 max-abs (`crates/t0-core/tests/batch.rs`, `#[ignore]`d, needs the checkpoint).

### Commands

```
# Export quantized weights
t0-cli export-gguf --weights <safetensors> --config <config.json> --quant q4_0 --out t0-alpha-q4_0.gguf

# Forecast/parity accept either format; --config is optional for GGUF (metadata is embedded)
t0-cli parity --fixtures fixtures --weights t0-alpha-q4_0.gguf --backend ndarray
t0-cli forecast --fixture 0 --weights <safetensors> --config <config.json> --backend wgpu

# Latency: median of 5 timed forward passes after 1 warm-up
t0-cli bench --weights t0-alpha-q4_0.gguf --backend ndarray --signals 1000 --context 512 --horizon 96 --reps 5
```

## Demo

`web/` offers all four released weights — alpha Q8_0/Q4_0 and beta Q8_0/Q4_0 — fetched from their Hugging Face repos (`worker.js`'s `MODELS` table); `LOCAL_MODELS_DIR` there can be pointed at a local `web/models/<key>/` instead, for testing before a repo is public.

## Publishing the demo

`tools/publish-pages.sh` builds both wasm engines (`fast` for WebGPU at `web/pkg-wgpu`, `ndarray` for CPU at `web/pkg`) and republishes committed HEAD's `web/` to an orphan `gh-pages` branch, following `../tts-web`'s layout (repo root = the served tree). Run it, then `git push origin gh-pages --force-with-lease`. GitHub Pages source: branch `gh-pages`, folder `/`.

## What's not done yet

- `FUTURE` covariate rows in the scaler, autoregressive rollout past 1024 steps, quantile interpolation/extrapolation for non-trained levels — see `docs/runs/2026-09-19-parity.md`'s last section.
- WGSL dequant/matmul kernels (Milestone 1a keeps quantization a load-time CPU-side dequant into f32 tensors; a real WASM+WebGPU deployment needs the on-GPU Q8/Q4 kernels described in `docs/reports/t0-alpha.md` §5 — deferred to the WASM/browser milestone).
- The WASM build itself (only `cargo build -p t0-core --target wasm32-unknown-unknown` is checked to compile; no browser page, no `web/` glue).
- The GIFT-Eval subset harness (accuracy verdict for Q8_0/Q4_0).
- The t0.run-style page and the METAR live metric (data layer).
- Reused nothing verbatim from `llm-web`: its `crates/llm-wasm/src/model.rs` is hardcoded to the `Wgpu` backend and the Qwen2 rotate-half RoPE convention (different from t0-alpha's interleaved-pair xpos RoPE), so a fresh backend-generic implementation was more direct than adapting it. The `getrandom/wasm_js` + `.cargo/config.toml` cfg-flag fix for the wasm32 build *was* copied from `llm-wasm/Cargo.toml`'s `web` feature.
