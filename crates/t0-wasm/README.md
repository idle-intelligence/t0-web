# t0-wasm

`wasm-bindgen` surface for the browser demo (`web/`) and the compare page
(`bench/compare/`), over either `t0-fast` (WebGPU) or `t0-core` (Burn, CPU).
Depends only on crates in this workspace.

## Build

Separate `wasm-pack` outputs, one per backend feature (backend is a
compile-time Cargo feature, same convention as `crates/cli`, never a
runtime switch inside the library). `scripts/build.sh` builds the two the
site uses:

```
wasm-pack build crates/t0-wasm --target web --out-dir pkg-fast --release --no-default-features --features fast
wasm-pack build crates/t0-wasm --target web --release
```

and places `pkg-fast/` (t0-fast, WebGPU) at `web/pkg-wgpu/` and `pkg/`
(ndarray, CPU) at `web/pkg/`. `web/worker.js` loads the WebGPU build when
the browser can create a WebGPU device, the CPU build otherwise. The
`wgpu` feature (Burn on WebGPU) still builds, but the site does not use it.

## API

- `initBackend(): Promise<void>` — **must be awaited once, before
  `T0Wasm.load`**, on every backend. On `wgpu` this drives the real async
  `requestAdapter`/`requestDevice` setup (`cubecl_wgpu::init_setup_async`);
  on `fast` it creates the `t0-fast` engine's device and rejects if none
  can be created; a no-op on `ndarray`. See "WebGPU on the Burn `wgpu`
  feature" below for why this exists. On `fast`, `forecast`,
  `forecastBatch` and `forecastBatchRows` reject with the engine's error
  instead of trapping.
- `T0Wasm.load(gguf_bytes: Uint8Array) -> T0Wasm` — parses a GGUF file (as
  written by `t0-cli export-gguf`) and builds the model.
- `instance.forecast(context: Float32Array, horizon: number) -> Promise<Float32Array>`
  — one forward pass, one signal. Always returns a `Promise` (JS:
  `await model.forecast(...)`) regardless of backend, since it always goes
  through `t0_core::forecast_async` (`into_data_async().await`, never the
  sync `into_data()` — see "WebGPU status"). Returns `horizon * nQuantiles()`
  values, time-major then quantile-minor (same layout as `t0-cli gifteval`'s
  output).
- `instance.nQuantiles() -> number` — 5 for t0-alpha, 21 for t0-beta
  (`config.quantile_levels.len()`).

## Backend

Compile-time Cargo feature: `ndarray` (default, CPU, `burn-ndarray`),
`wgpu` (`burn_wgpu::Wgpu<f32, i32>`) or `fast` (`t0-fast`), mutually
exclusive. `web/worker.js` selects which built `pkg` to load at runtime;
`forecast` and `initBackend` are async on every backend so the JS caller
never has to branch on which one it got.

## WebGPU on the Burn `wgpu` feature

Two hazards had to be fixed to get `wgpu` working in a real browser (not
just compiling for wasm32):

1. **Synchronous GPU readback deadlock.** `Tensor::into_data()` calls
   `try_read_sync`, which panics on wasm32 ("this can happen on platforms
   that don't support blocking futures like WASM") because there's no
   blocking executor. Fixed by adding `T0Model::forward_async` /
   `t0_core::forecast_async` / `forecast_series_async` (all in
   `crates/t0-core/src/model.rs`), which read back with
   `into_data_async().await` instead. `T0Wasm::forecast` always calls the
   `_async` path, on every backend — `NdArray`'s `into_data_async`
   resolves immediately (no real async work), `wgpu`'s does a genuine
   async GPU readback.
2. **Lazy synchronous device/adapter init.** Less obvious, and hit even
   *before* any forecast call, during `T0Wasm::load`: the first tensor op
   run on a `WgpuDevice` that hasn't been explicitly initialized triggers a
   lazy setup path that also panics with the same `try_read_sync` message,
   because `requestAdapter`/`requestDevice` are only available as async
   browser APIs. Fixed by exposing `initBackend()`, which awaits
   `cubecl_wgpu::init_setup_async::<WebGpu>(&WgpuDevice::default(), ..)`
   once before `T0Wasm.load` is ever called. `web/worker.js` awaits it
   right after the wasm module's own `default()` init. A no-op on
   `ndarray` (still safe/cheap to await so JS doesn't need to branch).

Verified in Playwright's bundled Chromium-for-Testing (`chromium-1229`,
`150.0.7871.24`) on an Apple M2: **no special launch flags were
needed** — `navigator.gpu` is present with a real hardware Metal adapter on
`http://` origins in this Chromium version by default (confirmed via
`adapter.features` listing GPU-specific features like `subgroups`,
`texture-compression-bc`; `navigator.gpu` was *not* present on
`about:blank`, which has an opaque origin). `--enable-unsafe-webgpu` /
`--use-angle=metal` were not needed and not passed — this Chromium-for-Testing
revision (150.0.7871.24) already ships WebGPU enabled by default over `http://`
origins on that machine; see `docs/runs/2026-09-19-web-smoke.md` for the exact
headless run and numbers.
