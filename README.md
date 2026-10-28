# t0-web

t0-alpha and t0-beta, The Forecasting Company's time-series foundation models, running in the browser: Rust compiled to WebAssembly, WebGPU inference with a CPU fallback, weights quantized to Q8_0 or Q4_0 GGUF.

[**Try the demo →**](https://idle-intelligence.github.io/t0-web/web/)

[**Try the compare page →**](https://idle-intelligence.github.io/t0-web/bench/compare/): the same engine next to the official ONNX INT8 export and, opt-in, an F32 reference, side by side in one tab. A run downloads one ONNX file and one GGUF: about 210 MB with Q8_0, 160 MB with Q4_0; turning on the F32 reference adds 407 MB more.

> **Disclaimer:** an independent port by idle-intelligence, not affiliated with or endorsed by The Forecasting Company. Weights are quantized from [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha) and [theforecastingcompany/t0-beta](https://huggingface.co/theforecastingcompany/t0-beta) (Apache-2.0); forecasts may differ slightly from the PyTorch implementation.

## Status

- F32 port matches the PyTorch reference to 1.6e-6 max-abs error for alpha and 3.2e-6 for beta, worst case across the native backends (CPU and Metal).
- Full GIFT-Eval protocol, 97 configs, normalized to Seasonal Naive: Q8_0 scores MASE 0.7258 / CRPS 0.4943 and Q4_0 0.7334 / 0.4973, against our F32 control at 0.7255 / 0.4942 and the published t0-alpha card at 0.7240 / 0.4941 (`docs/runs/2026-09-20-official-protocol-full97-normalized.md`).
- Quantized weights on Hugging Face: [alpha Q4_0](https://huggingface.co/idle-intelligence/t0-alpha-q4_0-webgpu) · [alpha Q8_0](https://huggingface.co/idle-intelligence/t0-alpha-q8_0-webgpu) · [beta Q4_0](https://huggingface.co/idle-intelligence/t0-beta-q4_0-webgpu) · [beta Q8_0](https://huggingface.co/idle-intelligence/t0-beta-q8_0-webgpu).
- Architecture research in `docs/reports/t0-alpha.md` and `docs/reports/t0-alpha-anatomy.md`.

## Layout

- `crates/t0-core`: the model, generic over `burn::tensor::backend::Backend`. Safetensors and GGUF loaders, RMSNorm, RoPE (interleaved-pair/xpos, matching `rotary_embedding_torch`), QK-norm multi-head attention (time + group/cross-variate), SwiGLU MLP, causal per-timestep scaler, patch encoder, quantile head. The reference path for parity checks, and the CPU engine in the browser.
- `crates/t0-fast`: the WebGPU engine the demo runs. Raw `wgpu` and hand-written WGSL kernels, no Burn at inference; Q8_0/Q4_0 weights stay resident on the GPU as GGUF blocks and are dequantized inside the matmul shaders; one command encoder and one readback per forecast.
- `crates/t0-wasm`: the `wasm-bindgen` surface. The `fast` feature builds it over `t0-fast` (WebGPU), the default `ndarray` feature over `t0-core` (CPU).
- `crates/cli`: `t0-cli`, native binary. `ndarray` feature (default), `wgpu` or `fast` picks the backend. Commands: `forecast`, `parity`, `bench`, `export-gguf`, `drift`, `gifteval`, `forecast-raw`.
- `web/`: the demo page (`index.html`) and its Web Worker (`worker.js`).
- `bench/compare/`: the compare page, this engine next to the official ONNX INT8 export in one tab.
- `scripts/build.sh`: builds both wasm engines and assembles the site.
- `tools/`: Python helpers for fixtures and GIFT-Eval scoring; `tools/make_fixtures.py` generates `fixtures/` from the reference PyTorch model.
- `fixtures/`: committed (small: ~430 KB), one manifest + per-case context/output/intermediate-activation binaries (little-endian f32).
- `docs/reports/`: architecture research and The Forecasting Company's published numbers.
- `docs/BENCHMARKS.md`, `docs/runs/`: research log.

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

Browser (both wasm engines and the site, into `_site/`):
```
scripts/build.sh
```
The wasm32 flags live in `.cargo/config.toml`: `+simd128`, and `--cfg getrandom_backend="wasm_js"` because Burn's dependency chain pulls in `rand`/`getrandom` even though this inference-only crate never calls it. The build fails if a built wasm loses `simd128`.

## Fixtures

`tools/make_fixtures.py` requires the reference PyTorch package:
```
python3 -m venv .venv && . .venv/bin/activate && pip install tfc-t0
python3 tools/make_fixtures.py
```
Python is used here only because the reference implementation is itself PyTorch: re-deriving golden outputs any other way would just be re-implementing the thing being checked, telling us nothing. This never runs as part of the Rust build; fixtures are committed.

## Parity gate

Needs the t0-alpha checkpoint (not committed; weights are never committed). `$MODELS_DIR` below is wherever you keep downloaded Hugging Face checkpoints (defaults to `models` under the repo if unset):
```
hf download theforecastingcompany/t0-alpha --local-dir $MODELS_DIR/hf/theforecastingcompany/t0-alpha
cargo run -p t0-cli --release -- parity \
  --fixtures fixtures \
  --weights $MODELS_DIR/hf/theforecastingcompany/t0-alpha/model.safetensors \
  --config $MODELS_DIR/hf/theforecastingcompany/t0-alpha/config.json
```
Gate: quantile output within 1e-4 max-abs of the reference on every fixture. Current result: 1.2e-6 (`docs/BENCHMARKS.md`).

The same checkpoint is needed for `cargo test -- --ignored` (`crates/t0-core/tests/{loader,parity}.rs`).

## Quantization

`t0-cli export-gguf` writes a GGUF v3 container (`crates/t0-core/src/gguf.rs`): llama.cpp's own block layouts, Q8_0 (32 values, f16 scale + 32 i8) and Q4_0 (32 values, f16 scale + 16 bytes of nibbles, matching `ggml_quantize_row_q4_0`'s exact rounding/sign convention), plus the model config as `t0.*` GGUF metadata keys, so a GGUF file is self-describing (no separate `config.json` needed to load one).

**Which tensors are quantized** (`Weights::is_quantizable` in `crates/t0-core/src/weights.rs`): the four big per-layer matmuls (`attention.wQKV.weight`, `attention.wO.weight`, `mlp.0.weight`, `mlp.2.weight`) to the requested Q8_0/Q4_0/f16 scheme. Everything else stays f16: patch encoder, type embeddings, decoder, every norm (`q_norm`/`k_norm`/`attention_block.norm`/layer `norm`/`out_norm`), and `head.quantile_levels`. Rationale: the patch encoder gates every downstream layer's accuracy, the decoder produces the quantile values directly, and norms/`quantile_levels` are tiny: quantizing them buys negligible size for real precision risk.

```
t0-cli export-gguf --weights <model.safetensors> --config <config.json> --quant f16|q8_0|q4_0 --out <file.gguf>
```

`t0-fast` keeps the four quantized matmuls as Q8_0/Q4_0 blocks in GPU buffers and dequantizes them inside the matmul shaders; the small f16 tensors are expanded to f32 at load. `t0-core` (the CPU engine, and the Burn `wgpu` backend) dequantizes every GGUF tensor to f32 at load, so there quantization shrinks the download, not the matmul dtype. Round-trip error against the PyTorch reference and the latency tables are in `docs/BENCHMARKS.md`; accuracy on the full GIFT-Eval protocol is in the Status section above.

## Native backends

`t0-cli --backend ndarray|wgpu|fast` picks the engine: `ndarray` and `wgpu` are Burn backends of `t0-core`, `fast` is `t0-fast` on native `wgpu`. Each is a Cargo feature of the `cli` crate (`cargo build -p t0-cli --release --no-default-features --features wgpu`, or `--features fast`), and `--backend` errors out when it does not match the binary's compiled feature: backend selection stays a compile-time choice, not a runtime switch. Burn `wgpu` F32 parity matches the PyTorch reference to the same order of magnitude as ndarray (worst case 1.6e-6 vs ndarray's 1.2e-6), and `t0-fast` to 1.25e-6.

## Batch path

`forecast_batch`/`TimeSeries::from_context_batch` run N independent signals in one forward pass by giving each its own group id; the existing cross-variate group-attention mask (group-id equality) isolates them. 8 batched copies of fixture 0 match 8x the single-signal forecast to <=1e-5 max-abs (`crates/t0-core/tests/batch.rs`, `#[ignore]`d, needs the checkpoint).

## Commands

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

`web/` offers all four released weights, alpha Q8_0/Q4_0 and beta Q8_0/Q4_0, fetched from their Hugging Face repos (`worker.js`'s `MODELS` table). The worker runs `t0-fast` on WebGPU when the browser can create a WebGPU device, and the `t0-core` CPU engine otherwise. `LOCAL_MODELS_DIR` in `worker.js` can point at a local `web/models/<key>/` instead of Hugging Face, for testing new weights.

## Publishing the demo

`scripts/build.sh` builds both wasm engines (`fast` for WebGPU at `web/pkg-wgpu`, `ndarray` for CPU at `web/pkg`) and assembles `_site/` with `web/` plus `bench/compare/index.html` and `bench/compare/compare.js` (no other assets: that page loads everything from Hugging Face and jsdelivr). `scripts/serve.py` serves `_site/` locally. The `pages` GitHub Actions workflow runs the same build on push to `main` and deploys `_site/` to GitHub Pages.

## Not done yet

- `FUTURE` covariate rows in the scaler (`crates/t0-core/src/scaler.rs` rejects them; only target and historical rows are ported).
