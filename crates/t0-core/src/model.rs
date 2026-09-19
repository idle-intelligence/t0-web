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
use crate::scaler::CausalScaler;
use crate::weights::Weights;

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
        })
    }

    /// `T0Forecaster.forward`: patch encoder -> transformer -> decoder ->
    /// quantile head. `series` must already be scaled (arcsinh'd) and
    /// patch-aligned. Returns `[v, p, patch_size, n_quantiles]` flattened
    /// row-major, plus (optionally) the patch-embedding / layer-0 trace.
    pub fn forward(&self, series: &TimeSeries, device: &B::Device, want_trace: bool) -> (Vec<f32>, Option<Trace<B>>) {
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
        let rope = RopeTables::<B>::new(p, cfg.head_dim(), device);

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
        let data = out.into_data();
        let raw: Vec<f32> = data.iter::<f32>().collect();

        let trace = patch_embedding_trace.map(|pe| Trace {
            patch_embedding: pe,
            layer0_output: layer0_output_trace.expect("layer0 trace set when want_trace is true"),
        });
        (raw, trace)
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

/// Batched forecast over `n_signals` independent univariate series (see
/// `TimeSeries::from_context_batch`): one forward pass, `v = n_signals`,
/// each signal isolated from the others by a distinct group id so
/// cross-variate group-attention layers never mix them.
pub fn forecast_batch<B: Backend>(
    model: &T0Model<B>,
    contexts: &[f32],
    n_signals: usize,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
) -> Vec<f32> {
    let raw = TimeSeries::from_context_batch(contexts, n_signals, t_ctx, horizon);
    forecast_series(model, raw, t_ctx, horizon, device, false).0
}

fn forecast_series<B: Backend>(
    model: &T0Model<B>,
    raw: TimeSeries,
    t_ctx: usize,
    horizon: usize,
    device: &B::Device,
    want_trace: bool,
) -> (Vec<f32>, Option<Trace<B>>) {
    let cfg = &model.config;
    let patch_size = cfg.patch_size;
    let v = raw.v;
    assert_eq!(t_ctx % patch_size, 0, "context_len must be patch-aligned for this milestone");
    assert_eq!(horizon % patch_size, 0, "horizon must be patch-aligned for this milestone");
    assert!(t_ctx + horizon <= 1024, "beyond max_horizon needs autoregressive rollout (unimplemented)");

    let window = raw.pad(patch_size, t_ctx); // no-op here since t_ctx is patch-aligned

    let scaler = CausalScaler::new(cfg.scaler_use_arcsinh, cfg.scaler_eps, &cfg.scaler_eps_mode);
    let (scaled, loc_scale) = scaler.scale_input(&window);

    let (mut predictions, trace) = model.forward(&scaled, device, want_trace);

    let n_patches = window.n_patches(patch_size);
    let n_q = cfg.n_quantiles();
    scaler.rescale_predictions(&mut predictions, &loc_scale, patch_size, n_patches, n_q);

    // Slice out the forecast region: patches [context_patches-1, +horizon_patches),
    // the decoder's next-patch prediction shifted by one (see model.rs's doc
    // comment and RolloutManager.predict_step).
    let context_patches = t_ctx / patch_size;
    let horizon_patches = horizon / patch_size;
    let per_patch = patch_size * n_q;
    let mut out = vec![0.0f32; v * horizon * n_q];
    for row in 0..v {
        for hp in 0..horizon_patches {
            let src_patch = context_patches - 1 + hp;
            let src_base = row * n_patches * per_patch + src_patch * per_patch;
            let dst_base = row * horizon * n_q + hp * per_patch;
            out[dst_base..dst_base + per_patch].copy_from_slice(&predictions[src_base..src_base + per_patch]);
        }
    }
    (out, trace)
}
