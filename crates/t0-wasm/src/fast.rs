//! Same JS surface as `burn_impl.rs` (`T0Wasm`, `initBackend`), backed by
//! `t0-fast`'s raw-wgpu engine instead of a Burn model. `initBackend` does
//! the real async `requestAdapter`/`requestDevice` work here (there's no
//! separate lazy-init hazard to dodge like Burn's `cubecl_wgpu` -- this
//! crate never touches a device until `Engine::new_async` explicitly
//! builds one), so `T0Wasm::load` needs that already-built `Engine`; JS
//! must still call `await initBackend()` once before `T0Wasm.load(...)`,
//! same calling convention as the Burn variant.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

fn to_js_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

thread_local! {
    // wasm32-unknown-unknown is single-threaded, so a thread_local Rc (not
    // Arc) is enough; Rc::clone is a cheap refcount bump, letting each
    // async method hold its own handle across an `.await` without an
    // unsafe pointer or re-borrowing the cell while suspended.
    static ENGINE: RefCell<Option<Rc<t0_fast::Engine>>> = const { RefCell::new(None) };
}

fn engine() -> Rc<t0_fast::Engine> {
    ENGINE.with(|cell| cell.borrow().clone()).expect("initBackend() must be awaited before T0Wasm.load()")
}

#[wasm_bindgen(js_name = initBackend)]
pub async fn init_backend() {
    let engine = t0_fast::Engine::new_async().await.expect("t0-fast: no WebGPU adapter/device");
    ENGINE.with(|cell| {
        // Idempotent: a second `initBackend()` call (e.g. a page that
        // reloads the model) just keeps the first engine/device.
        cell.borrow_mut().get_or_insert_with(|| Rc::new(engine));
    });
}

#[wasm_bindgen]
pub struct T0Wasm {
    model: t0_fast::GpuModel,
}

#[wasm_bindgen]
impl T0Wasm {
    /// Parses a GGUF byte buffer (as exported by `t0-cli export-gguf`) and
    /// uploads every tensor straight to `wgpu::Buffer`s -- no Burn tensor
    /// at any point. `--fast-quant` isn't exposed here yet: this always
    /// loads F32-resident weights (see `docs/runs/2026-09-20-perf.md`'s
    /// Phase C note on wiring Q8_0/Q4_0 through to the wasm surface as a
    /// follow-up).
    #[wasm_bindgen]
    pub fn load(gguf_bytes: &[u8]) -> Result<T0Wasm, JsValue> {
        console_error_panic_hook::set_once();
        let (weights, config) = t0_core::Weights::load_gguf_bytes(gguf_bytes).map_err(to_js_err)?;
        let model = t0_fast::load_model(&engine(), &weights, config, t0_fast::WeightQuant::F32).map_err(to_js_err)?;
        Ok(T0Wasm { model })
    }

    #[wasm_bindgen(js_name = nQuantiles)]
    pub fn n_quantiles(&self) -> usize {
        self.model.config.n_quantiles()
    }

    /// One forward pass, one signal (`v = 1`). Same contract as the Burn
    /// variant's `forecast` (see `burn_impl.rs`): raw context in, `horizon
    /// * n_quantiles` values out, time-major then quantile-minor.
    #[wasm_bindgen]
    pub async fn forecast(&self, context: Vec<f32>, horizon: usize) -> Vec<f32> {
        let t_ctx = context.len();
        let engine = engine();
        t0_fast::forecast_async(&engine, &self.model, &context, 1, t_ctx, horizon).await.expect("t0-fast forecast failed")
    }

    /// `n_signals` independent forecasts, chunked at `chunk_size` (0 means
    /// "no chunking") -- same contract as the Burn variant's
    /// `forecastBatch`.
    #[wasm_bindgen(js_name = forecastBatch)]
    pub async fn forecast_batch(&self, context: Vec<f32>, n_signals: usize, horizon: usize, chunk_size: usize) -> Vec<f32> {
        let t_ctx = context.len();
        let mut contexts = Vec::with_capacity(n_signals * t_ctx);
        for _ in 0..n_signals {
            contexts.extend_from_slice(&context);
        }
        let engine = engine();
        t0_fast::forecast_batch_chunked_async(&engine, &self.model, &contexts, n_signals, t_ctx, horizon, chunk_size)
            .await
            .expect("t0-fast forecast_batch failed")
    }
}
