//! `t0-wasm`: `wasm-bindgen` surface over `t0-core`, self-contained (no
//! `llm-web` dependency). CPU-only (`burn-ndarray`) for now — see
//! `README.md` for the WebGPU status. `NdArray` is fully synchronous, so
//! `forecast`'s `Tensor::into_data()` call inside `t0_core::model::forward`
//! never touches an async GPU readback path; that constraint only applies
//! once a `wgpu` backend is wired in here (tracked in `README.md`).

use wasm_bindgen::prelude::*;

#[cfg(feature = "ndarray")]
type Backend = burn_ndarray::NdArray<f32>;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type Backend = burn_wgpu::Wgpu<f32, i32>;

fn to_js_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
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
    /// the native CLI (`t0_core::model::forecast_series`).
    #[wasm_bindgen]
    pub fn forecast(&self, context: &[f32], horizon: usize) -> Vec<f32> {
        let device = Default::default();
        let t_ctx = context.len();
        let (out, _) = t0_core::forecast(&self.model, context, 1, t_ctx, horizon, &device, false);
        out
    }
}
