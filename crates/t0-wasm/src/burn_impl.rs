//! `t0-wasm`: `wasm-bindgen` surface over `t0-core`, self-contained (no
//! `llm-web` dependency). Backend is a compile-time Cargo feature, same
//! convention as `crates/cli` (`ndarray` default, `wgpu` mutually
//! exclusive with it, see `Cargo.toml`).
//!
//! `forecast` awaits `t0_core::forecast_async` (never the sync `forecast`,
//! which calls `Tensor::into_data()` — a deadlock hazard on `wgpu` in a
//! real browser, see this crate's `README.md`) so JS can always
//! `await model.forecast(...)` regardless of which backend the build was
//! made with: `NdArray`'s `into_data_async` resolves immediately (no real
//! async work), `wgpu`'s does a genuine async GPU readback. `#[wasm_bindgen]`
//! turns the `async fn` into a method that returns a JS `Promise`.

use wasm_bindgen::prelude::*;

#[cfg(feature = "ndarray")]
type Backend = burn_ndarray::NdArray<f32>;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type Backend = burn_wgpu::Wgpu<f32, i32>;

fn to_js_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Must be awaited once, before `T0Wasm::load`, on every backend. On
/// `wgpu` this does the real work: `cubecl_wgpu`'s device/adapter/queue
/// setup (`requestAdapter`/`requestDevice`) is only available as an async
/// API in the browser (no blocking executor exists in WASM), so it has to
/// be driven from JS's event loop via `init_setup_async` *before* any
/// tensor touches `WgpuDevice::default()` — otherwise the first tensor op
/// falls back to a synchronous lazy-init path
/// (`cubecl_common::reader::try_read_sync`) and panics with "Failed to
/// read tensor data synchronously... this can happen on platforms that
/// don't support blocking futures like WASM" (confirmed by hitting exactly
/// that panic before this function existed). On `ndarray` this is a no-op
/// (`NdArray` has no async device setup) but is still safe/cheap to await
/// so JS doesn't need to branch on backend.
#[wasm_bindgen(js_name = initBackend)]
pub async fn init_backend() {
    #[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
    {
        let device = burn_wgpu::WgpuDevice::default();
        burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, Default::default()).await;
    }
}

#[wasm_bindgen]
pub struct T0Wasm {
    model: t0_core::T0Model<Backend>,
}

#[wasm_bindgen]
impl T0Wasm {
    /// Parses a GGUF byte buffer (as exported by `t0-cli export-gguf`) and
    /// builds the model. Two-phase in spirit: `Weights::load_gguf_bytes`
    /// dequantizes into a plain `HashMap<String, Vec<f32>>` first, then
    /// `T0Model::load` copies each tensor onto the backend device and the
    /// intermediate `Weights` is dropped when this function returns.
    #[wasm_bindgen]
    pub fn load(gguf_bytes: &[u8]) -> Result<T0Wasm, JsValue> {
        console_error_panic_hook::set_once();
        let (weights, config) = t0_core::Weights::load_gguf_bytes(gguf_bytes).map_err(to_js_err)?;
        let device = Default::default();
        let model = t0_core::T0Model::load(&weights, config, &device).map_err(to_js_err)?;
        Ok(T0Wasm { model })
    }

    /// Number of trained quantile levels (5 for t0-alpha/t0-beta) — the
    /// caller uses this to interpret `forecast`'s flat output as
    /// `[horizon, n_quantiles]`, quantile-minor (matches `t0-cli`'s own
    /// output layout, see `forecast_series` in `t0-core/src/model.rs`).
    #[wasm_bindgen(js_name = nQuantiles)]
    pub fn n_quantiles(&self) -> usize {
        self.model.config.n_quantiles()
    }

    /// One forward pass, one signal (`v = 1`). `context` is the raw
    /// (unscaled) time series; `horizon` is the number of future steps
    /// requested. Returns `horizon * n_quantiles` values, time-major then
    /// quantile-minor. No autoregressive rollout — same
    /// `context.len() + horizon <= 1024` (padded to patch size) ceiling as
    /// the native CLI (`t0_core::model::forecast_series`). Returns a
    /// `Promise` (JS: `await model.forecast(...)`) on every backend, since
    /// this always goes through `t0_core::forecast_async` — see this
    /// module's doc comment.
    #[wasm_bindgen]
    pub async fn forecast(&self, context: Vec<f32>, horizon: usize) -> Vec<f32> {
        let device = Default::default();
        let t_ctx = context.len();
        let (out, _) = t0_core::forecast_async(&self.model, &context, 1, t_ctx, horizon, &device, false).await;
        out
    }

    /// `n_signals` independent forecasts in one call, each over the same
    /// `context` (used for batch-latency benchmarking, see
    /// `bench/ours/index.html`), internally chunked at `chunk_size` (0 means
    /// "no chunking") the same way `t0-cli bench --chunk` is on native — see
    /// `t0_core::forecast_batch_chunked_async`'s doc comment for why the
    /// chunk cap exists on `wgpu`. Returns `n_signals * horizon * n_quantiles`
    /// values, signal-major then time-major then quantile-minor.
    #[wasm_bindgen(js_name = forecastBatch)]
    pub async fn forecast_batch(&self, context: Vec<f32>, n_signals: usize, horizon: usize, chunk_size: usize) -> Vec<f32> {
        let device = Default::default();
        let t_ctx = context.len();
        let mut contexts = Vec::with_capacity(n_signals * t_ctx);
        for _ in 0..n_signals {
            contexts.extend_from_slice(&context);
        }
        t0_core::forecast_batch_chunked_async(&self.model, &contexts, n_signals, t_ctx, horizon, &device, chunk_size).await
    }
}
