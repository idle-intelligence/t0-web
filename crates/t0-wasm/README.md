# t0-wasm

`wasm-bindgen` surface over `t0-core` for the one-signal browser instrument
(`web/`). Self-contained: only depends on `t0-core` (path dependency in
this workspace), never on `llm-web`.

## Build

```
wasm-pack build crates/t0-wasm --target web --release
```

Produces `crates/t0-wasm/pkg/` (`t0_wasm.js` + `.wasm`), loaded by
`web/worker.js` as an ES module.

## API

- `T0Wasm.load(gguf_bytes: Uint8Array) -> T0Wasm` — parses a GGUF file (as
  written by `t0-cli export-gguf`) and builds the model.
- `instance.forecast(context: Float32Array, horizon: number) -> Float32Array`
  — one forward pass, one signal. Returns `horizon * nQuantiles()` values,
  time-major then quantile-minor (same layout as `t0-cli gifteval`'s
  output).
- `instance.nQuantiles() -> number` — 5 for both t0-alpha and t0-beta
  (`config.quantile_levels.len()`).

## Backend

CPU only (`burn-ndarray`, the `ndarray` feature, on by default). `NdArray`
is fully synchronous, so `Tensor::into_data()` inside
`t0_core::model::T0Model::forward` never touches an async GPU readback path
here — that's a `wgpu`-only hazard (`.into_data_async().await` is required
there, never `.into_data()`), not one this crate has yet.

## WebGPU status: not wired up

A `wgpu` Cargo feature exists (`Backend = burn_wgpu::Wgpu<f32, i32>`) and
the crate *compiles* under it for wasm32, because `t0-core` already builds
against any `burn::tensor::backend::Backend` and `burn-wgpu` compiles for
wasm32 (used natively already, see `crates/cli`'s `wgpu` feature and
`docs/BENCHMARKS.md`'s Metal rows). It has **not been run** in a browser:

- `Weights::load`/`load_gguf_bytes` build every tensor with
  `Tensor::from_floats`, which is a synchronous host->device upload; fine
  on `wgpu` too (uploads don't need the async-readback treatment, only
  *downloads* do), but untested end-to-end here.
- `forecast`'s `Tensor::into_data()` call in `t0_core::model::forward`
  (`model.rs:187`) is a **synchronous GPU readback**. On `wgpu` in a real
  browser this can deadlock (the constraint this repo's CLAUDE.md and the
  owning agent's brief both call out: never `.into_data()` on `wgpu` in
  WASM, always `.into_data_async().await`). `t0-core::forecast` is a sync
  function today (used synchronously by the native CLI too), so making the
  `wgpu` path safe means either (a) an async `forecast_async` variant in
  `t0-core` behind a feature, or (b) restricting `wgpu` use in `t0-wasm` to
  a Web Worker with a wrapper that awaits `into_data_async()` at the
  crate's own boundary instead of inside `t0-core`.
- This session's hard constraint was CPU-only measurements (the GPU was
  occupied by a concurrent fine-tune) — no wgpu/WebGPU build or run was
  attempted here, in the browser or otherwise.

TODO (next session, GPU free): add `t0_core::model::forecast_async` behind
a `wgpu`-shaped feature, swap `T0Wasm::forecast` to await it when built
with `--features wgpu`, verify in Playwright with a real (non-headless-CPU)
WebGPU-capable Chromium, and only then advertise a `wgpu` row in
`docs/BENCHMARKS.md`.
