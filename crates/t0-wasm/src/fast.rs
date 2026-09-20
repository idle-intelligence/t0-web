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
    /// at any point, and no F32-expanded copy of the big per-layer matmuls
    /// either: `load_model_from_gguf` keeps their Q8_0/Q4_0 block bytes
    /// exactly as they arrive off the wire (the page already downloads a
    /// quantized GGUF), reading straight into the packed-u32/scale buffers
    /// the dequant kernels use. Only the small always-F16-on-disk tensors
    /// (patch encoder, decoder, norms, type embeddings) get dequantized to
    /// f32 here, same as the Burn path.
    #[wasm_bindgen]
    pub fn load(gguf_bytes: &[u8]) -> Result<T0Wasm, JsValue> {
        console_error_panic_hook::set_once();
        let model = t0_fast::load_model_from_gguf(&engine(), gguf_bytes).map_err(to_js_err)?;
        Ok(T0Wasm { model })
    }

    #[wasm_bindgen(js_name = nQuantiles)]
    pub fn n_quantiles(&self) -> usize {
        self.model.config.n_quantiles()
    }

    /// Sum of every GPU buffer this instance holds: persistent weights
    /// (`GpuModel::total_weight_bytes`) plus the per-forward `Pool`
    /// working set (`Engine::pool::resident_bytes`, 0 before the first
    /// `forecast`/`forecastBatch` call). For the browser GPU-memory report.
    #[wasm_bindgen(js_name = gpuBytes)]
    pub fn gpu_bytes(&self) -> f64 {
        (self.model.total_weight_bytes() + engine().pool.resident_bytes()) as f64
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

    /// `n_signals` independent forecasts over *distinct* contexts (possibly
    /// different lengths), unlike `forecastBatch`'s single-context
    /// replication. `contexts` is every row's values concatenated
    /// back-to-back, `lengths[i]` gives row `i`'s length (so
    /// `contexts.len() == sum(lengths)`, no caller-side padding needed).
    /// Chunked at `BATCH_ROWS_CHUNK` rows per GPU batch forward pass; see
    /// `t0_fast::forecast_batch_rows_chunked_async`'s doc comment for why
    /// each row's result matches a standalone `forecast()` call on that
    /// row exactly.
    #[wasm_bindgen(js_name = forecastBatchRows)]
    pub async fn forecast_batch_rows(&self, contexts: Vec<f32>, lengths: Vec<u32>, horizon: usize) -> Vec<f32> {
        const BATCH_ROWS_CHUNK: usize = 24;
        let lengths: Vec<usize> = lengths.into_iter().map(|l| l as usize).collect();
        let engine = engine();
        t0_fast::forecast_batch_rows_chunked_async(&engine, &self.model, &contexts, &lengths, horizon, BATCH_ROWS_CHUNK)
            .await
            .expect("t0-fast forecast_batch_rows failed")
    }
}
