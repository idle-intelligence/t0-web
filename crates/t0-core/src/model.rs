//! `t0.model.model.T0Forecaster` / `t0.model.layers.transformer.Transformer`
//! assembled from `ops.rs` primitives, plus the single-window predict driver
//! (`t0.model.rollout.RolloutManager.predict_step`, restricted to
//! `prediction_length <= max_horizon` — no autoregressive rollout, which is
//! all three of this milestone's fixtures need; see
//! `docs/reports/t0-alpha.md` §5).

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::{LayerType, T0Config};
use crate::data::{MaskType, TimeSeries};
use crate::mask::{build_group_mask, build_time_mask, patch_attendable, reduce_patch_metadata};
use crate::ops::{mhsa, quantile_head, residual_block, rmsnorm, swiglu_ffn, RopeTables};
use crate::quantile::{interpolate_quantiles, prob_mass, weighted_quantile};
use crate::scaler::CausalScaler;
use crate::weights::Weights;

/// `T0Forecaster`'s `max_horizon`: the model predicts this many steps in one
/// forward pass; beyond it, `RolloutManager` (`t0/model/rollout.py`) falls
/// back to the autoregressive rollout that `forecast_rollout` below ports.
/// Both t0-alpha and t0-beta train with `max_horizon = 1024` (see
/// `docs/reports/t0-alpha.md`); this isn't in `T0Config` because neither
/// published checkpoint varies it.
pub const MAX_HORIZON: usize = 1024;

struct AttnWeights<B: Backend> {
    norm_scale: Tensor<B, 1>,
    w_qkv: Tensor<B, 2>,
    b_qkv: Tensor<B, 1>,
    w_o: Tensor<B, 2>,
    b_o: Tensor<B, 1>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
}

struct LayerWeights<B: Backend> {
    ty: LayerType,
    attn: AttnWeights<B>,
    mlp_norm_scale: Tensor<B, 1>,
    w0: Tensor<B, 2>,
    b0: Tensor<B, 1>,
    w2: Tensor<B, 2>,
    b2: Tensor<B, 1>,
}

struct ResidualBlockWeights<B: Backend> {
    hidden_w: Tensor<B, 2>,
    hidden_b: Tensor<B, 1>,
    output_w: Tensor<B, 2>,
    output_b: Tensor<B, 1>,
    residual_w: Tensor<B, 2>,
    residual_b: Tensor<B, 1>,
}

pub struct T0Model<B: Backend> {
    pub config: T0Config,
    patch_encoder: ResidualBlockWeights<B>,
    type_embeddings: Tensor<B, 2>, // [3, embed_dim]
    layers: Vec<LayerWeights<B>>,
    out_norm_scale: Tensor<B, 1>,
    decoder: ResidualBlockWeights<B>,
    // RoPE tables are a pure function of (patch count `p`, head_dim) -- not
    // of any per-call series data -- so they're safe to cache across
    // `forward_tensor` calls keyed by `p`. Repeated forecasts at a fixed
    // context/horizon (the common case: one model, one series length,
    // called in a loop) hit this every time after the first, skipping 3
    // `Tensor::from_floats` GPU buffer writes/forecast (`ops.rs`'s
    // `RopeTables::new`) -- each buffer write forces cubecl-wgpu's stream to
    // flush before it (`ScheduleTask::Write` in cubecl-wgpu's `stream.rs`),
    // so this also cuts 3 of the ~7 per-forecast forced command-buffer
    // flushes on the wgpu backend (see `docs/runs/2026-09-20-perf.md`).
    rope_cache: std::cell::RefCell<Option<(usize, RopeTables<B>)>>,
}

/// Intermediate activations for parity debugging (`docs/reports/t0-alpha.md`
/// asks for the patch embedding and first-layer output).
pub struct Trace<B: Backend> {
    pub patch_embedding: Tensor<B, 3>, // [v, p, embed_dim]
    pub layer0_output: Tensor<B, 3>,   // [v, p, embed_dim]
}

fn load_residual_block<B: Backend>(w: &Weights, prefix: &str, device: &B::Device) -> Result<ResidualBlockWeights<B>> {
    Ok(ResidualBlockWeights {
        hidden_w: w.get2(&format!("{prefix}.mlp.hidden_layer.weight"), device)?,
        hidden_b: w.get1(&format!("{prefix}.mlp.hidden_layer.bias"), device)?,
        output_w: w.get2(&format!("{prefix}.mlp.output_layer.weight"), device)?,
        output_b: w.get1(&format!("{prefix}.mlp.output_layer.bias"), device)?,
        residual_w: w.get2(&format!("{prefix}.residual_layer.weight"), device)?,
        residual_b: w.get1(&format!("{prefix}.residual_layer.bias"), device)?,
    })
}

fn apply_residual_block<B: Backend>(x: Tensor<B, 2>, w: &ResidualBlockWeights<B>) -> Tensor<B, 2> {
    residual_block(
        x,
        w.hidden_w.clone(),
        w.hidden_b.clone(),
        w.output_w.clone(),
        w.output_b.clone(),
        w.residual_w.clone(),
        w.residual_b.clone(),
    )
}

impl<B: Backend> T0Model<B> {
    pub fn load(weights: &Weights, config: T0Config, device: &B::Device) -> Result<Self> {
        let patch_encoder = load_residual_block(weights, "patch_encoder.projection", device)?;
        let type_embeddings = weights.get2("patch_encoder.type_embeddings.weight", device)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, ty) in config.layer_types().into_iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            let attn = AttnWeights {
                norm_scale: weights.get1(&format!("{p}.attention_block.norm.scale"), device)?,
                w_qkv: weights.get2(&format!("{p}.attention_block.attention.wQKV.weight"), device)?,
                b_qkv: weights.get1(&format!("{p}.attention_block.attention.wQKV.bias"), device)?,
                w_o: weights.get2(&format!("{p}.attention_block.attention.wO.weight"), device)?,
                b_o: weights.get1(&format!("{p}.attention_block.attention.wO.bias"), device)?,
                q_norm: weights.get1(&format!("{p}.attention_block.attention.q_norm.scale"), device)?,
                k_norm: weights.get1(&format!("{p}.attention_block.attention.k_norm.scale"), device)?,
            };
            layers.push(LayerWeights {
                ty,
                attn,
                mlp_norm_scale: weights.get1(&format!("{p}.norm.scale"), device)?,
                w0: weights.get2(&format!("{p}.mlp.0.weight"), device)?,
                b0: weights.get1(&format!("{p}.mlp.0.bias"), device)?,
                w2: weights.get2(&format!("{p}.mlp.2.weight"), device)?,
                b2: weights.get1(&format!("{p}.mlp.2.bias"), device)?,
            });
        }

        let out_norm_scale = weights.get1("transformer.out_norm.scale", device)?;
        let decoder = load_residual_block(weights, "decoder", device)?;

        Ok(T0Model {
            config,
            patch_encoder,
            type_embeddings,
            layers,
            out_norm_scale,
            decoder,
            rope_cache: std::cell::RefCell::new(None),
        })
    }

    /// `T0Forecaster.forward`: patch encoder -> transformer -> decoder ->
    /// quantile head. `series` must already be scaled (arcsinh'd) and
    /// patch-aligned. Returns `[v, p, patch_size, n_quantiles]` flattened
    /// row-major, plus (optionally) the patch-embedding / layer-0 trace.
    ///
    /// Synchronous GPU readback (`Tensor::into_data()`): fine on native
    /// wgpu/Metal and on the CPU `ndarray` backend, but a deadlock hazard on
    /// `wgpu` in a real browser (WebGPU readback is only ever async there).
    /// Never call this from WASM when built against `wgpu` — use
    /// `forward_async` instead (see `crates/t0-wasm/README.md`).
    pub fn forward(&self, series: &TimeSeries, device: &B::Device, want_trace: bool) -> (Vec<f32>, Option<Trace<B>>) {
        let (out, trace) = self.forward_tensor(series, device, want_trace);
        let data = out.into_data();
        let raw: Vec<f32> = data.iter::<f32>().collect();
        (raw, trace)
    }

    /// Same as `forward`, but reads the final tensor back with
    /// `into_data_async().await` instead of the synchronous `into_data()`.
    /// This is the one that's safe to call from `t0-wasm` when built with
    /// the `wgpu` feature for a browser target.
    pub async fn forward_async(&self, series: &TimeSeries, device: &B::Device, want_trace: bool) -> (Vec<f32>, Option<Trace<B>>) {
        let (out, trace) = self.forward_tensor(series, device, want_trace);
        let data = out.into_data_async().await.expect("GPU readback failed");
        let raw: Vec<f32> = data.iter::<f32>().collect();
        (raw, trace)
    }

    fn forward_tensor(&self, series: &TimeSeries, device: &B::Device, want_trace: bool) -> (Tensor<B, 1>, Option<Trace<B>>) {
        let cfg = &self.config;
        let v = series.v;
        let patch_size = cfg.patch_size;
        let p = series.n_patches(patch_size);
        let embed = cfg.embed_dim;

        // --- patch encoder ---
        let mut concat = vec![0.0f32; v * p * patch_size * 3];
        for row in 0..v {
            for patch in 0..p {
                for k in 0..patch_size {
                    let t_col = patch * patch_size + k;
                    let src = row * series.t + t_col;
                    let base = (row * p + patch) * (patch_size * 3);
                    concat[base + k] = series.variates[src];
                    concat[base + patch_size + k] = k as f32 / patch_size as f32;
                    concat[base + 2 * patch_size + k] = if series.mask[src] == MaskType::Valid as i8 { 1.0 } else { 0.0 };
                }
            }
        }
        let concat_t: Tensor<B, 2> = Tensor::<B, 1>::from_floats(concat.as_slice(), device).reshape([v * p, patch_size * 3]);
        let projected = apply_residual_block(concat_t, &self.patch_encoder).reshape([v, p, embed]);

        let patched_variate_type = reduce_patch_metadata(&series.variate_type, &series.mask, v, patch_size);
        let patched_group_ids = reduce_patch_metadata(&series.group_ids, &series.mask, v, patch_size);
        let attendable = patch_attendable(&series.mask, v, patch_size);

        let type_idx: Vec<i64> = patched_variate_type.iter().map(|t| (*t).max(0)).collect();
        let type_emb = gather_rows(self.type_embeddings.clone(), &type_idx, device).reshape([v, p, embed]);
        let mut x = projected + type_emb;
        let patch_embedding_trace = if want_trace { Some(x.clone()) } else { None };

        // --- masks + RoPE, shared across all layers (constant seq_len = p) ---
        let time_mask_flat = build_time_mask(&patched_group_ids, &patched_variate_type, &attendable, v, p);
        let time_mask: Tensor<B, 4> = Tensor::<B, 1>::from_floats(time_mask_flat.as_slice(), device).reshape([v, 1, p, p]);
        let group_mask_flat = build_group_mask(&patched_group_ids, v, p);
        let group_mask: Tensor<B, 4> = Tensor::<B, 1>::from_floats(group_mask_flat.as_slice(), device).reshape([p, 1, v, v]);
        let rope = {
            let mut cache = self.rope_cache.borrow_mut();
            if !matches!(&*cache, Some((cached_p, _)) if *cached_p == p) {
                *cache = Some((p, RopeTables::<B>::new(p, cfg.head_dim(), device)));
            }
            cache.as_ref().expect("just populated above").1.clone()
        };

        let mut layer0_output_trace = None;
        for (i, layer) in self.layers.iter().enumerate() {
            x = self.forward_layer(x, layer, &time_mask, &group_mask, &rope, cfg.num_heads);
            if want_trace && i == 0 {
                layer0_output_trace = Some(x.clone());
            }
        }
        x = rmsnorm(x, self.out_norm_scale.clone(), 1e-8);

        // --- decoder + quantile head ---
        let n_q = cfg.n_quantiles();
        let decoded = apply_residual_block(x.reshape([v * p, embed]), &self.decoder); // [v*p, patch_size*n_q]
        let decoded = decoded.reshape([v * p * patch_size, n_q]);
        let out = quantile_head(decoded, n_q);
        let out: Tensor<B, 1> = out.reshape([v * p * patch_size * n_q]);

        let trace = patch_embedding_trace.map(|pe| Trace {
            patch_embedding: pe,
            layer0_output: layer0_output_trace.expect("layer0 trace set when want_trace is true"),
        });
        (out, trace)
    }

    fn forward_layer(
        &self,
        x: Tensor<B, 3>,
        layer: &LayerWeights<B>,
        time_mask: &Tensor<B, 4>,
        group_mask: &Tensor<B, 4>,
        rope: &RopeTables<B>,
        num_heads: usize,
    ) -> Tensor<B, 3> {
        let normed = rmsnorm(x.clone(), layer.attn.norm_scale.clone(), 1e-8);
        let attn_out = match layer.ty {
            LayerType::Time => mhsa(
                normed,
                layer.attn.w_qkv.clone(),
                layer.attn.b_qkv.clone(),
                layer.attn.w_o.clone(),
                layer.attn.b_o.clone(),
                layer.attn.q_norm.clone(),
                layer.attn.k_norm.clone(),
                num_heads,
                time_mask.clone(),
                Some(rope),
            ),
            LayerType::Group => {
                let flipped = normed.swap_dims(0, 1); // [v,p,e] -> [p,v,e]
                let out = mhsa(
                    flipped,
                    layer.attn.w_qkv.clone(),
                    layer.attn.b_qkv.clone(),
                    layer.attn.w_o.clone(),
                    layer.attn.b_o.clone(),
                    layer.attn.q_norm.clone(),
                    layer.attn.k_norm.clone(),
                    num_heads,
                    group_mask.clone(),
                    None,
                );
                out.swap_dims(0, 1) // back to [v,p,e]
            }
        };
        let x = x + attn_out;

        let [v, p, e] = x.dims();
        let mlp_normed = rmsnorm(x.clone(), layer.mlp_norm_scale.clone(), 1e-8).reshape([v * p, e]);
        let mlp_out = swiglu_ffn(
            mlp_normed,
            layer.w0.clone(),
            layer.b0.clone(),
            layer.w2.clone(),
            layer.b2.clone(),
            self.config.mlp_hidden_dim,
        )
        .reshape([v, p, e]);
        x + mlp_out
    }
}

/// Embedding lookup as a one-hot matmul (`one_hot[n, rows] @ table[rows, cols]`)
/// so this stays a device-side op with no synchronous readback — the
/// `type_embeddings` table only has 3 rows, so this is cheap.
fn gather_rows<B: Backend>(table: Tensor<B, 2>, idx: &[i64], device: &B::Device) -> Tensor<B, 2> {
    let [rows, _cols] = table.dims();
    let mut one_hot = vec![0.0f32; idx.len() * rows];
    for (n, &i) in idx.iter().enumerate() {
        one_hot[n * rows + i as usize] = 1.0;
    }
    let one_hot: Tensor<B, 2> = Tensor::<B, 1>::from_floats(one_hot.as_slice(), device).reshape([idx.len(), rows]);
    one_hot.matmul(table)
}

/// Full `predict()` for a single window (no autoregressive rollout — valid
/// whenever `context_len + horizon <= max_horizon` (1024), true for every
/// fixture in this milestone). Mirrors
/// `RolloutManager.{prepare_rollout_buffer,predict_step}` with
/// `query_quantile_levels == model.head.quantile_levels` (so
/// `interpolate_quantiles` is the identity — see
/// `docs/reports/t0-alpha.md` §5 and `t0/quantile.py`).
pub fn forecast<B: Backend>(
    model: &T0Model<B>,
    context: &[f32],
    v: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    want_trace: bool,
) -> (Vec<f32>, Option<Trace<B>>) {
    let raw = TimeSeries::from_context(context, v, t_ctx, horizon);
    forecast_series(model, raw, t_ctx, horizon, device, want_trace)
}

/// Same as `forecast`, but via `T0Model::forward_async` (`into_data_async().await`
/// readback) — the one safe to call from `t0-wasm`'s `wgpu` feature in a
/// browser. See `T0Model::forward_async`'s doc comment.
pub async fn forecast_async<B: Backend>(
    model: &T0Model<B>,
    context: &[f32],
    v: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    want_trace: bool,
) -> (Vec<f32>, Option<Trace<B>>) {
    let raw = TimeSeries::from_context(context, v, t_ctx, horizon);
    forecast_series_async(model, raw, t_ctx, horizon, device, want_trace).await
}

/// Default chunk size for `forecast_batch`'s wgpu-safe chunked execution.
/// `cubek-matmul`'s autotune picks a group-attention matmul tile that needs
/// 40 KB of shared memory on shapes with `n_signals > 16` (`v`/`n_signals`
/// is the group-attention matmul's contraction dim); Metal's per-threadgroup
/// limit is 32 KB (see `docs/runs/2026-09-19-milestone1a.md`). 16 is the
/// largest `n_signals` confirmed to fit under that limit — see the binary
/// search recorded there.
pub const DEFAULT_BATCH_CHUNK: usize = 16;

/// Batched forecast over `n_signals` independent univariate series (see
/// `TimeSeries::from_context_batch`): one forward pass, `v = n_signals`,
/// each signal isolated from the others by a distinct group id so
/// cross-variate group-attention layers never mix them.
///
/// Internally chunks into groups of `DEFAULT_BATCH_CHUNK` signals (see
/// `forecast_batch_chunked` to override the chunk size) so the group-attention
/// matmul's contraction dim never exceeds the shape that trips the wgpu
/// shared-memory autotune bug above. Transparent to callers: output is
/// identical to one un-chunked forward pass (each chunk gets fresh group
/// ids and the chunks are fully independent — no cross-chunk attention is
/// possible even in the un-chunked path).
pub fn forecast_batch<B: Backend>(
    model: &T0Model<B>,
    contexts: &[f32],
    n_signals: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
) -> Vec<f32> {
    forecast_batch_chunked(model, contexts, n_signals, t_ctx, horizon, device, DEFAULT_BATCH_CHUNK)
}

/// `forecast_batch` with an overridable chunk size (0 or >= n_signals means
/// "no chunking, one forward pass").
pub fn forecast_batch_chunked<B: Backend>(
    model: &T0Model<B>,
    contexts: &[f32],
    n_signals: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    chunk_size: usize,
) -> Vec<f32> {
    let chunk_size = if chunk_size == 0 { n_signals } else { chunk_size.min(n_signals) };
    let n_q = model.config.n_quantiles();
    let mut out = vec![0.0f32; n_signals * horizon * n_q];
    let mut start = 0;
    while start < n_signals {
        let n = chunk_size.min(n_signals - start);
        let chunk_ctx = &contexts[start * t_ctx..(start + n) * t_ctx];
        let raw = TimeSeries::from_context_batch(chunk_ctx, n, t_ctx, horizon);
        let (chunk_out, _) = forecast_series(model, raw, t_ctx, horizon, device, false);
        out[start * horizon * n_q..(start + n) * horizon * n_q].copy_from_slice(&chunk_out);
        start += n;
    }
    out
}

/// Same as `forecast_batch_chunked`, but via `forecast_series_async`
/// (`into_data_async().await` readback) — the one safe to call from
/// `t0-wasm`'s `wgpu` feature in a browser, same reasoning as
/// `forecast_async` above.
pub async fn forecast_batch_chunked_async<B: Backend>(
    model: &T0Model<B>,
    contexts: &[f32],
    n_signals: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    chunk_size: usize,
) -> Vec<f32> {
    let chunk_size = if chunk_size == 0 { n_signals } else { chunk_size.min(n_signals) };
    let n_q = model.config.n_quantiles();
    let mut out = vec![0.0f32; n_signals * horizon * n_q];
    let mut start = 0;
    while start < n_signals {
        let n = chunk_size.min(n_signals - start);
        let chunk_ctx = &contexts[start * t_ctx..(start + n) * t_ctx];
        let raw = TimeSeries::from_context_batch(chunk_ctx, n, t_ctx, horizon);
        let (chunk_out, _) = forecast_series_async(model, raw, t_ctx, horizon, device, false).await;
        out[start * horizon * n_q..(start + n) * horizon * n_q].copy_from_slice(&chunk_out);
        start += n;
    }
    out
}

/// Shared pre-`model.forward`/`forward_async` setup for `forecast_series`
/// and `forecast_series_async`: patch-align padding + causal scaling.
/// Returns the scaled+padded window, the scaler's loc/scale (for
/// rescaling predictions afterward), and the padded context/horizon sizes.
struct PreparedWindow {
    window: TimeSeries,
    loc_scale: crate::scaler::LocScale,
    padded_t_ctx: usize,
    padded_horizon: usize,
}

fn forecast_series_prepare<B: Backend>(
    model: &T0Model<B>,
    raw: TimeSeries,
    t_ctx: usize,
    horizon: usize,
) -> (TimeSeries, PreparedWindow) {
    let cfg = &model.config;
    let patch_size = cfg.patch_size;

    // GIFT-Eval windows aren't patch-aligned (e.g. horizon=30 for daily
    // series, horizon=8 for weekly) — round context/horizon up to whole
    // patches via `TimeSeries::pad` (left-pads context, right-pads horizon
    // with WITHHELD), forward once, then truncate the padded-horizon output
    // back down to the caller's real `horizon`. Matches `Patcher.pad` +
    // `RolloutManager.predict_step`'s own pad/truncate, restricted (like
    // the rest of this milestone) to windows that fit in one non-rollout
    // forward pass.
    let pad_left = (patch_size - t_ctx % patch_size) % patch_size;
    let padded_horizon = horizon.div_ceil(patch_size) * patch_size;
    let padded_t_ctx = t_ctx + pad_left;
    assert!(
        padded_horizon <= MAX_HORIZON,
        "single-window forecast_series can't exceed max_horizon ({MAX_HORIZON}) in one pass -- use forecast_rollout"
    );

    let window = raw.pad(patch_size, t_ctx);

    let scaler = CausalScaler::new(cfg.scaler_use_arcsinh, cfg.scaler_eps, &cfg.scaler_eps_mode);
    let (scaled, loc_scale) = scaler.scale_input(&window);

    (
        scaled,
        PreparedWindow { window, loc_scale, padded_t_ctx, padded_horizon },
    )
}

/// Shared post-`model.forward`/`forward_async` finish for `forecast_series`
/// and `forecast_series_async`: rescale predictions back out of the
/// causal-scaled space, then slice out the forecast region.
fn forecast_series_finish<B: Backend>(
    model: &T0Model<B>,
    mut predictions: Vec<f32>,
    trace: Option<Trace<B>>,
    prepared: &PreparedWindow,
    v: usize,
    horizon: usize,
) -> (Vec<f32>, Option<Trace<B>>) {
    let PreparedWindow { window, loc_scale, padded_t_ctx, padded_horizon } = prepared;
    let padded_t_ctx = *padded_t_ctx;
    let padded_horizon = *padded_horizon;
    let cfg = &model.config;
    let patch_size = cfg.patch_size;
    let scaler = CausalScaler::new(cfg.scaler_use_arcsinh, cfg.scaler_eps, &cfg.scaler_eps_mode);
    let n_patches = window.n_patches(patch_size);
    let n_q = cfg.n_quantiles();
    scaler.rescale_predictions(&mut predictions, loc_scale, patch_size, n_patches, n_q);

    // Slice out the forecast region: patches [context_patches-1, +horizon_patches),
    // the decoder's next-patch prediction shifted by one (see model.rs's doc
    // comment and RolloutManager.predict_step), then drop the padded tail
    // steps beyond the caller's real `horizon`.
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
        return (padded_out, trace);
    }
    let mut out = vec![0.0f32; v * horizon * n_q];
    for row in 0..v {
        let src = row * padded_horizon * n_q;
        let dst = row * horizon * n_q;
        out[dst..dst + horizon * n_q].copy_from_slice(&padded_out[src..src + horizon * n_q]);
    }
    (out, trace)
}

fn forecast_series<B: Backend>(
    model: &T0Model<B>,
    raw: TimeSeries,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    want_trace: bool,
) -> (Vec<f32>, Option<Trace<B>>) {
    let v = raw.v;
    let (scaled, prepared) = forecast_series_prepare(model, raw, t_ctx, horizon);
    let (predictions, trace) = model.forward(&scaled, device, want_trace);
    forecast_series_finish(model, predictions, trace, &prepared, v, horizon)
}

/// `RolloutManager.predict`, restricted to a single univariate target series
/// with no known-future covariates -- the shape every GIFT-Eval entry takes
/// (`T0Predictor._contexts_from_entries` drops covariates for the same
/// reason: "the benchmark scores the target channel only", `t0/evaluation/
/// predictor.py`). That drops `TimeSeries`/mask/group-id bookkeeping
/// entirely: a "buffer" is just a growing `Vec<f32>` of real-then-synthetic
/// values, and `time_slice(decoded, context_width+decoded+horizon)` is
/// "take the trailing `context_width` elements" of that vector.
///
/// Feed-back rule (`RolloutManager.update_buffer_with_predictions` +
/// `expand_prediction_paths`): beyond `max_horizon`, one path per query
/// quantile level is rolled forward independently, each path feeding back
/// *its own* column of the previously *reduced* (query-level) prediction --
/// not its own raw model output -- as if that quantile trajectory were the
/// observed continuation. Each step's raw per-path, per-trained-quantile
/// outputs are then mixed back down to the query levels by
/// `QuantileRolloutReducer.reduce` (`weighted_quantile` over the
/// trained-level x query-level probability-mass grid) before being fed back
/// again -- so per-path divergence within one AR block is real (each path
/// samples its own belief), but is re-collapsed to a single distribution at
/// every block boundary.
///
/// Context re-windowing to `context_length` (e.g. 8192): every AR step's
/// window is the trailing `context_width = round_up(context_length,
/// patch_size)` elements of the (real + fed-back) buffer -- oldest
/// already-decoded steps fall off the left as new ones are appended on the
/// right, so the model always attends over the same-sized window, never a
/// growing one (`docs/runs/2026-09-19-gifteval-subset.md`'s 512-context cap
/// is the special case where the whole series fits and nothing ever falls
/// off).
///
/// `query_quantile_levels` must be sorted ascending and strictly inside
/// `(0, 1)` -- extrapolation past the model's trained quantile range
/// (`extrapolate_quantiles` in `t0/quantile.py`) isn't implemented; GIFT-Eval's
/// query levels (deciles) are always inside t0-alpha/t0-beta's trained range.
pub fn forecast_rollout<B: Backend>(
    model: &T0Model<B>,
    context: &[f32],
    t_ctx: usize,
    prediction_length: usize,
    query_quantile_levels: &[f32],
    device: &B::Device,
) -> Vec<f32> {
    let patch_size = model.config.patch_size;
    let trained = &model.config.quantile_levels;
    let n_trained = trained.len();
    let trained_mass = prob_mass(trained);
    let n_query = query_quantile_levels.len();
    let query_mass = prob_mass(query_quantile_levels);
    let round_up = |x: usize, m: usize| x.div_ceil(m) * m;
    let context_width = round_up(t_ctx, patch_size);

    let horizon0 = round_up(prediction_length, patch_size).min(MAX_HORIZON);
    let (block0, _) = forecast(model, context, 1, t_ctx, horizon0, device, false);
    let mut first: Vec<f32> = Vec::with_capacity(horizon0 * n_query);
    for h in 0..horizon0 {
        first.extend(interpolate_quantiles(query_quantile_levels, trained, &block0[h * n_trained..(h + 1) * n_trained]));
    }
    if prediction_length <= horizon0 {
        first.truncate(prediction_length * n_query);
        return first;
    }

    let n_paths = n_query;
    let mut paths_context: Vec<Vec<f32>> = vec![context.to_vec(); n_paths];
    let mut remaining = prediction_length - horizon0;
    let mut prev_block = first.clone();
    let mut prev_width = horizon0;
    let mut out_chunks: Vec<Vec<f32>> = vec![first];

    while remaining > 0 {
        for (j, buf) in paths_context.iter_mut().enumerate() {
            for h in 0..prev_width {
                buf.push(prev_block[h * n_query + j]);
            }
        }
        let horizon = round_up(remaining, patch_size).min(MAX_HORIZON);
        let mut batch_ctx = vec![0.0f32; n_paths * context_width];
        for (j, buf) in paths_context.iter().enumerate() {
            let start = buf.len() - context_width.min(buf.len());
            let window = &buf[start..];
            let dst = &mut batch_ctx[j * context_width..(j + 1) * context_width];
            dst[context_width - window.len()..].copy_from_slice(window);
        }
        let blocks = forecast_batch(model, &batch_ctx, n_paths, context_width, horizon, device); // [n_paths, horizon, n_trained]

        let mut reduced = Vec::with_capacity(horizon * n_query);
        let mut samples = vec![0.0f32; n_paths * n_trained];
        let mut weights = vec![0.0f32; n_paths * n_trained];
        for j in 0..n_paths {
            for pq in 0..n_trained {
                weights[j * n_trained + pq] = trained_mass[pq] * query_mass[j];
            }
        }
        for h in 0..horizon {
            for j in 0..n_paths {
                for pq in 0..n_trained {
                    samples[j * n_trained + pq] = blocks[j * horizon * n_trained + h * n_trained + pq];
                }
            }
            reduced.extend(weighted_quantile(query_quantile_levels, &weights, &samples));
        }

        prev_width = horizon;
        prev_block = reduced.clone();
        out_chunks.push(reduced);
        remaining -= horizon;
    }
    let mut out: Vec<f32> = out_chunks.concat();
    out.truncate(prediction_length * n_query);
    out
}

/// Same as `forecast_series`, but via `T0Model::forward_async`.
async fn forecast_series_async<B: Backend>(
    model: &T0Model<B>,
    raw: TimeSeries,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    want_trace: bool,
) -> (Vec<f32>, Option<Trace<B>>) {
    let v = raw.v;
    let (scaled, prepared) = forecast_series_prepare(model, raw, t_ctx, horizon);
    let (predictions, trace) = model.forward_async(&scaled, device, want_trace).await;
    forecast_series_finish(model, predictions, trace, &prepared, v, horizon)
}
