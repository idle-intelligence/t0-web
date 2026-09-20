//! Second forward path for `t0-alpha`: no Burn at inference. `engine.rs`
//! owns a raw `wgpu::Device`/`Queue` and ten hand-written WGSL compute
//! pipelines; `model.rs` records one forecast's whole dispatch chain into a
//! single `wgpu::CommandEncoder`, submits once, and reads back once. Burn
//! (`t0-core`) stays the loader (via `Weights::get_raw`) and the reference
//! path for parity checks. See `docs/reports/t0-alpha.md` for the
//! architecture this ports, and `docs/runs/2026-09-20-perf.md` for why:
//! Burn's per-`linear()`-call dispatch overhead (not FLOPs) is ~85% of one
//! forecast's wall time on this backend.

pub mod engine;
pub mod model;
pub mod pool;
pub mod quant;
mod rope_tables;

use anyhow::Result;
use t0_core::data::TimeSeries;
use t0_core::scaler::CausalScaler;
use t0_core::{T0Config, Weights};

pub use engine::Engine;
pub use model::GpuModel;
pub use quant::WeightQuant;

/// Same default as `t0_core::DEFAULT_BATCH_CHUNK` -- kept as a separate
/// constant (not reused from t0-core) since it documents a Burn/cubecl
/// wgpu-autotune limitation that doesn't apply to this crate's own
/// hand-written kernels (no shared-memory autotune at all here); 16 is
/// carried over only so a batch-24 bench is chunked the same way as the
/// Burn baseline for a fair comparison, not because this crate needs it.
pub const DEFAULT_BATCH_CHUNK: usize = 16;

pub fn load_model(engine: &Engine, weights: &Weights, config: T0Config, quant: WeightQuant) -> Result<GpuModel> {
    GpuModel::load(engine, weights, config, quant)
}

/// Shared core of `forecast_async`/`forecast_batch_chunked_async`: pad,
/// scale, forward, rescale, and slice the forecast region back out of one
/// already-built `TimeSeries` window. Mirrors
/// `t0_core::model::forecast_series{,_prepare,_finish}` exactly.
async fn forecast_window_async(engine: &Engine, model: &GpuModel, raw: TimeSeries, t_ctx: usize, horizon: usize) -> Result<Vec<f32>> {
    let cfg = &model.config;
    let patch_size = cfg.patch_size;
    let v = raw.v;

    let pad_left = (patch_size - t_ctx % patch_size) % patch_size;
    let padded_horizon = horizon.div_ceil(patch_size) * patch_size;
    let padded_t_ctx = t_ctx + pad_left;
    assert!(padded_t_ctx + padded_horizon <= 1024, "beyond max_horizon needs autoregressive rollout (unimplemented)");

    let window = raw.pad(patch_size, t_ctx);

    let scaler = CausalScaler::new(cfg.scaler_use_arcsinh, cfg.scaler_eps, &cfg.scaler_eps_mode);
    let (scaled, loc_scale) = scaler.scale_input(&window);

    let mut predictions = model::forward_async(engine, model, &scaled).await?;

    let n_patches = window.n_patches(patch_size);
    let n_q = cfg.n_quantiles();
    scaler.rescale_predictions(&mut predictions, &loc_scale, patch_size, n_patches, n_q);

    let context_patches = padded_t_ctx / patch_size;
    let horizon_patches = padded_horizon / patch_size;
    let per_patch = patch_size * n_q;
    let mut padded_out = vec![0.0f32; v * padded_horizon * n_q];
    for row in 0..v {
        for hp in 0..horizon_patches {
            let src_patch = context_patches - 1 + hp;
            let src_base = row * n_patches * per_patch + src_patch * per_patch;
            let dst_base = row * padded_horizon * n_q + hp * per_patch;
            padded_out[dst_base..dst_base + per_patch].copy_from_slice(&predictions[src_base..src_base + per_patch]);
        }
    }
    if padded_horizon == horizon {
        return Ok(padded_out);
    }
    let mut out = vec![0.0f32; v * horizon * n_q];
    for row in 0..v {
        let src = row * padded_horizon * n_q;
        let dst = row * horizon * n_q;
        out[dst..dst + horizon * n_q].copy_from_slice(&padded_out[src..src + horizon * n_q]);
    }
    Ok(out)
}

/// Single (possibly multivariate/joint) series -- `v` rows share one group
/// id, so group-attention layers cross-attend between them (matches
/// `t0_core::forecast`'s `TimeSeries::from_context`).
pub async fn forecast_async(engine: &Engine, model: &GpuModel, context: &[f32], v: usize, t_ctx: usize, horizon: usize) -> Result<Vec<f32>> {
    forecast_window_async(engine, model, TimeSeries::from_context(context, v, t_ctx, horizon), t_ctx, horizon).await
}

#[cfg(not(target_arch = "wasm32"))]
pub fn forecast(engine: &Engine, model: &GpuModel, context: &[f32], v: usize, t_ctx: usize, horizon: usize) -> Result<Vec<f32>> {
    pollster::block_on(forecast_async(engine, model, context, v, t_ctx, horizon))
}

/// `n_signals` independent univariate series (matches
/// `t0_core::forecast_batch_chunked`'s `TimeSeries::from_context_batch`:
/// each signal gets a distinct group id, so group-attention layers never
/// mix them even though they share one forward pass / one command
/// encoder). Chunked so `chunk_size` bounds the group-attention `seq`
/// (`attention.wgsl`'s `MAX_SEQ=128` -- see model.rs); 0 or
/// `>= n_signals` means "no chunking, one forward pass".
pub async fn forecast_batch_chunked_async(
    engine: &Engine,
    model: &GpuModel,
    contexts: &[f32],
    n_signals: usize,
    t_ctx: usize,
    horizon: usize,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let chunk_size = if chunk_size == 0 { n_signals } else { chunk_size.min(n_signals) };
    let n_q = model.config.n_quantiles();
    let mut out = vec![0.0f32; n_signals * horizon * n_q];
    let mut start = 0;
    while start < n_signals {
        let n = chunk_size.min(n_signals - start);
        let chunk_ctx = &contexts[start * t_ctx..(start + n) * t_ctx];
        let raw = TimeSeries::from_context_batch(chunk_ctx, n, t_ctx, horizon);
        let chunk_out = forecast_window_async(engine, model, raw, t_ctx, horizon).await?;
        out[start * horizon * n_q..(start + n) * horizon * n_q].copy_from_slice(&chunk_out);
        start += n;
    }
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn forecast_batch_chunked(engine: &Engine, model: &GpuModel, contexts: &[f32], n_signals: usize, t_ctx: usize, horizon: usize, chunk_size: usize) -> Result<Vec<f32>> {
    pollster::block_on(forecast_batch_chunked_async(engine, model, contexts, n_signals, t_ctx, horizon, chunk_size))
}
