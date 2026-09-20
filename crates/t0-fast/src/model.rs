//! GPU-resident `T0Model` forward pass with no Burn tensor anywhere: every
//! op in `t0_core::ops` (rmsnorm, linear, RoPE, mhsa, swiglu_ffn,
//! quantile_head) has a hand-written WGSL counterpart in `engine.rs`'s
//! pipeline set, and `forward_async` records every dispatch for one
//! forecast into a single `wgpu::CommandEncoder`, submits once, and reads
//! back once (`Engine::read_buffer`). CPU-side prep (patch concat, masks,
//! the causal scaler, RoPE table math) is reused as-is from `t0_core` --
//! none of that touches a GPU buffer until the single upload per input.

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::BindGroupEntry;

use t0_core::config::LayerType;
use t0_core::data::{MaskType, TimeSeries};
use t0_core::mask::{build_group_mask, build_time_mask, patch_attendable, reduce_patch_metadata};
use t0_core::{T0Config, Weights};

use crate::engine::Engine;
use crate::rope_tables::RopeTables;

const HEAD_DIM: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddDims {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherDims {
    rows: u32,
    embed: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RmsFullDims {
    rows: u32,
    dim: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RmsQkDims {
    rows: u32,
    heads: u32,
    head_dim: u32,
    base_offset: u32,
    row_stride: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeDims {
    rows: u32,
    heads: u32,
    head_dim: u32,
    base_offset: u32,
    row_stride: u32,
    seq_len: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnDims {
    outer: u32,
    seq: u32,
    heads: u32,
    embed: u32,
    qkv_stride: u32,
    mask_outer_stride: u32,
    scale: f32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeDims {
    a: u32,
    b: u32,
    e: u32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SiluDims {
    rows: u32,
    hidden: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuantileDims {
    rows: u32,
    n_q: u32,
    _p0: u32,
    _p1: u32,
}

struct ResidualBlockBuf {
    hidden_w: wgpu::Buffer,
    hidden_b: wgpu::Buffer,
    output_w: wgpu::Buffer,
    output_b: wgpu::Buffer,
    residual_w: wgpu::Buffer,
    residual_b: wgpu::Buffer,
    in_dim: u32,
    hidden_dim: u32,
    out_dim: u32,
}

struct LayerBuf {
    ty: LayerType,
    norm_scale: wgpu::Buffer,
    w_qkv: wgpu::Buffer,
    b_qkv: wgpu::Buffer,
    w_o: wgpu::Buffer,
    b_o: wgpu::Buffer,
    q_norm: wgpu::Buffer,
    k_norm: wgpu::Buffer,
    mlp_norm_scale: wgpu::Buffer,
    w0: wgpu::Buffer,
    b0: wgpu::Buffer,
    w2: wgpu::Buffer,
    b2: wgpu::Buffer,
    mlp_hidden: u32,
}

pub struct GpuModel {
    pub config: T0Config,
    patch_encoder: ResidualBlockBuf,
    type_embeddings: wgpu::Buffer,
    layers: Vec<LayerBuf>,
    out_norm_scale: wgpu::Buffer,
    decoder: ResidualBlockBuf,
}

fn load_residual_block(engine: &Engine, weights: &Weights, prefix: &str) -> Result<ResidualBlockBuf> {
    let (hw_shape, hw) = weights.get_raw(&format!("{prefix}.mlp.hidden_layer.weight"))?;
    let (_, hb) = weights.get_raw(&format!("{prefix}.mlp.hidden_layer.bias"))?;
    let (ow_shape, ow) = weights.get_raw(&format!("{prefix}.mlp.output_layer.weight"))?;
    let (_, ob) = weights.get_raw(&format!("{prefix}.mlp.output_layer.bias"))?;
    let (rw_shape, rw) = weights.get_raw(&format!("{prefix}.residual_layer.weight"))?;
    let (_, rb) = weights.get_raw(&format!("{prefix}.residual_layer.bias"))?;
    let in_dim = hw_shape[1] as u32;
    let hidden_dim = hw_shape[0] as u32;
    let out_dim = ow_shape[0] as u32;
    assert_eq!(rw_shape[1] as u32, in_dim);
    assert_eq!(rw_shape[0] as u32, out_dim);
    Ok(ResidualBlockBuf {
        hidden_w: engine.buf_f32(hw, "hidden_w"),
        hidden_b: engine.buf_f32(hb, "hidden_b"),
        output_w: engine.buf_f32(ow, "output_w"),
        output_b: engine.buf_f32(ob, "output_b"),
        residual_w: engine.buf_f32(rw, "residual_w"),
        residual_b: engine.buf_f32(rb, "residual_b"),
        in_dim,
        hidden_dim,
        out_dim,
    })
}

impl GpuModel {
    pub fn load(engine: &Engine, weights: &Weights, config: T0Config) -> Result<Self> {
        let patch_encoder = load_residual_block(engine, weights, "patch_encoder.projection")?;
        let (_, type_emb) = weights.get_raw("patch_encoder.type_embeddings.weight")?;
        let type_embeddings = engine.buf_f32(type_emb, "type_embeddings");

        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, ty) in config.layer_types().into_iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            let (_, norm_scale) = weights.get_raw(&format!("{p}.attention_block.norm.scale"))?;
            let (_, w_qkv) = weights.get_raw(&format!("{p}.attention_block.attention.wQKV.weight"))?;
            let (_, b_qkv) = weights.get_raw(&format!("{p}.attention_block.attention.wQKV.bias"))?;
            let (_, w_o) = weights.get_raw(&format!("{p}.attention_block.attention.wO.weight"))?;
            let (_, b_o) = weights.get_raw(&format!("{p}.attention_block.attention.wO.bias"))?;
            let (_, q_norm) = weights.get_raw(&format!("{p}.attention_block.attention.q_norm.scale"))?;
            let (_, k_norm) = weights.get_raw(&format!("{p}.attention_block.attention.k_norm.scale"))?;
            let (_, mlp_norm_scale) = weights.get_raw(&format!("{p}.norm.scale"))?;
            let (w0_shape, w0) = weights.get_raw(&format!("{p}.mlp.0.weight"))?;
            let (_, b0) = weights.get_raw(&format!("{p}.mlp.0.bias"))?;
            let (_, w2) = weights.get_raw(&format!("{p}.mlp.2.weight"))?;
            let (_, b2) = weights.get_raw(&format!("{p}.mlp.2.bias"))?;
            let mlp_hidden = (w0_shape[0] / 2) as u32;
            layers.push(LayerBuf {
                ty,
                norm_scale: engine.buf_f32(norm_scale, "norm_scale"),
                w_qkv: engine.buf_f32(w_qkv, "w_qkv"),
                b_qkv: engine.buf_f32(b_qkv, "b_qkv"),
                w_o: engine.buf_f32(w_o, "w_o"),
                b_o: engine.buf_f32(b_o, "b_o"),
                q_norm: engine.buf_f32(q_norm, "q_norm"),
                k_norm: engine.buf_f32(k_norm, "k_norm"),
                mlp_norm_scale: engine.buf_f32(mlp_norm_scale, "mlp_norm_scale"),
                w0: engine.buf_f32(w0, "w0"),
                b0: engine.buf_f32(b0, "b0"),
                w2: engine.buf_f32(w2, "w2"),
                b2: engine.buf_f32(b2, "b2"),
                mlp_hidden,
            });
        }

        let (_, out_norm_scale) = weights.get_raw("transformer.out_norm.scale")?;
        let out_norm_scale = engine.buf_f32(out_norm_scale, "out_norm_scale");
        let decoder = load_residual_block(engine, weights, "decoder")?;

        Ok(GpuModel {
            config,
            patch_encoder,
            type_embeddings,
            layers,
            out_norm_scale,
            decoder,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn linear(
    engine: &Engine,
    encoder: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    rows: u32,
    in_dim: u32,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    out_dim: u32,
    act: u32,
) -> wgpu::Buffer {
    let out = engine.buf_empty((rows * out_dim) as usize, "linear_out");
    let dims = engine.buf_uniform(
        LinearDims {
            m: rows,
            k: in_dim,
            n: out_dim,
            act,
        },
        "linear_dims",
    );
    let bg = engine.bind_group(
        &engine.linear,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.linear, &bg, (out_dim.div_ceil(16), rows.div_ceil(16), 1), "linear");
    out
}

fn add_inplace(engine: &Engine, encoder: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) {
    let dims = engine.buf_uniform(AddDims { len, _p0: 0, _p1: 0, _p2: 0 }, "add_dims");
    let bg = engine.bind_group(
        &engine.add_inplace,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.add_inplace, &bg, (len.div_ceil(256), 1, 1), "add_inplace");
}

fn rmsnorm_full(engine: &Engine, encoder: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, scale: &wgpu::Buffer, rows: u32, dim: u32) -> wgpu::Buffer {
    let out = engine.buf_empty((rows * dim) as usize, "rmsnorm_out");
    let dims = engine.buf_uniform(RmsFullDims { rows, dim, _p0: 0, _p1: 0 }, "rmsnorm_dims");
    let bg = engine.bind_group(
        &engine.rmsnorm_full,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm_full, &bg, (rows.div_ceil(64), 1, 1), "rmsnorm_full");
    out
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm_qk(
    engine: &Engine,
    encoder: &mut wgpu::CommandEncoder,
    buf: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    rows: u32,
    heads: u32,
    base_offset: u32,
    row_stride: u32,
) {
    let dims = engine.buf_uniform(
        RmsQkDims {
            rows,
            heads,
            head_dim: HEAD_DIM,
            base_offset,
            row_stride,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        },
        "rmsnorm_qk_dims",
    );
    let bg = engine.bind_group(
        &engine.rmsnorm_qk,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm_qk, &bg, ((rows * heads).div_ceil(64), 1, 1), "rmsnorm_qk");
}

#[allow(clippy::too_many_arguments)]
fn rope(
    engine: &Engine,
    encoder: &mut wgpu::CommandEncoder,
    buf: &wgpu::Buffer,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    rows: u32,
    heads: u32,
    base_offset: u32,
    row_stride: u32,
    seq_len: u32,
) {
    let dims = engine.buf_uniform(
        RopeDims {
            rows,
            heads,
            head_dim: HEAD_DIM,
            base_offset,
            row_stride,
            seq_len,
            _p0: 0,
            _p1: 0,
        },
        "rope_dims",
    );
    let bg = engine.bind_group(
        &engine.rope,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    let half = HEAD_DIM / 2;
    engine.dispatch(encoder, &engine.rope, &bg, ((rows * heads * half).div_ceil(64), 1, 1), "rope");
}

#[allow(clippy::too_many_arguments)]
fn attention(
    engine: &Engine,
    encoder: &mut wgpu::CommandEncoder,
    qkv: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    outer: u32,
    seq: u32,
    heads: u32,
    embed: u32,
) -> wgpu::Buffer {
    assert!(seq <= 64, "t0-fast Phase A attention kernel caps seq_len at 64 (MAX_SEQ in attention.wgsl), got {seq}");
    let qkv_stride = 3 * embed;
    let out = engine.buf_empty((outer * seq * embed) as usize, "attn_out");
    let dims = engine.buf_uniform(
        AttnDims {
            outer,
            seq,
            heads,
            embed,
            qkv_stride,
            mask_outer_stride: seq * seq,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            _p0: 0,
        },
        "attn_dims",
    );
    let bg = engine.bind_group(
        &engine.attention,
        &[
            BindGroupEntry { binding: 0, resource: qkv.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: mask.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.attention, &bg, (outer * heads, 1, 1), "attention");
    out
}

fn transpose_outer(engine: &Engine, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, a: u32, b: u32, e: u32) -> wgpu::Buffer {
    let dst = engine.buf_empty((a * b * e) as usize, "transpose_out");
    let dims = engine.buf_uniform(TransposeDims { a, b, e, _p0: 0 }, "transpose_dims");
    let bg = engine.bind_group(
        &engine.transpose_outer,
        &[
            BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.transpose_outer, &bg, ((a * b).div_ceil(64), 1, 1), "transpose_outer");
    dst
}

fn silu_mul(engine: &Engine, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, rows: u32, hidden: u32) -> wgpu::Buffer {
    let out = engine.buf_empty((rows * hidden) as usize, "silu_out");
    let dims = engine.buf_uniform(SiluDims { rows, hidden, _p0: 0, _p1: 0 }, "silu_dims");
    let bg = engine.bind_group(
        &engine.silu_mul,
        &[
            BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.silu_mul, &bg, ((rows * hidden).div_ceil(256), 1, 1), "silu_mul");
    out
}

fn residual_block(engine: &Engine, encoder: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, rows: u32, w: &ResidualBlockBuf) -> wgpu::Buffer {
    let hidden = linear(engine, encoder, x, rows, w.in_dim, &w.hidden_w, &w.hidden_b, w.hidden_dim, 1);
    let out = linear(engine, encoder, &hidden, rows, w.hidden_dim, &w.output_w, &w.output_b, w.out_dim, 0);
    let residual = linear(engine, encoder, x, rows, w.in_dim, &w.residual_w, &w.residual_b, w.out_dim, 0);
    add_inplace(engine, encoder, &out, &residual, rows * w.out_dim);
    out
}

/// Mirrors `t0_core::model::T0Model::forward_tensor`, one GPU dispatch per
/// `ops.rs` primitive, all recorded into a single encoder. Returns the raw
/// `[v, p, patch_size, n_quantiles]` flattened predictions -- same contract
/// as `T0Model::forward_async`.
pub async fn forward_async(engine: &Engine, model: &GpuModel, series: &TimeSeries) -> Result<Vec<f32>> {
    let cfg = &model.config;
    let v = series.v as u32;
    let patch_size = cfg.patch_size as u32;
    let p = series.n_patches(cfg.patch_size) as u32;
    let embed = cfg.embed_dim as u32;
    let heads = cfg.num_heads as u32;
    let rows = v * p;

    if embed / heads != HEAD_DIM {
        return Err(anyhow!("t0-fast Phase A hardcodes head_dim=64, config gives {}", embed / heads));
    }

    // --- patch encoder input (host-side concat, identical to model.rs) ---
    let mut concat = vec![0.0f32; (rows * patch_size * 3) as usize];
    for row in 0..series.v {
        for patch in 0..p as usize {
            for k in 0..cfg.patch_size {
                let t_col = patch * cfg.patch_size + k;
                let src = row * series.t + t_col;
                let base = (row * p as usize + patch) * (cfg.patch_size * 3);
                concat[base + k] = series.variates[src];
                concat[base + cfg.patch_size + k] = k as f32 / cfg.patch_size as f32;
                concat[base + 2 * cfg.patch_size + k] = if series.mask[src] == MaskType::Valid as i8 { 1.0 } else { 0.0 };
            }
        }
    }

    let patched_variate_type = reduce_patch_metadata(&series.variate_type, &series.mask, series.v, cfg.patch_size);
    let patched_group_ids = reduce_patch_metadata(&series.group_ids, &series.mask, series.v, cfg.patch_size);
    let attendable = patch_attendable(&series.mask, series.v, cfg.patch_size);
    let type_idx: Vec<u32> = patched_variate_type.iter().map(|t| (*t).max(0) as u32).collect();

    let time_mask_flat = build_time_mask(&patched_group_ids, &patched_variate_type, &attendable, series.v, p as usize);
    let group_mask_flat = build_group_mask(&patched_group_ids, series.v, p as usize);
    let rope_tables = RopeTables::new(p as usize, HEAD_DIM as usize);

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t0-fast forward") });

    let concat_buf = engine.buf_f32(&concat, "concat");
    let x = residual_block(engine, &mut encoder, &concat_buf, rows, &model.patch_encoder);

    let idx_buf = engine.buf_u32(&type_idx, "type_idx");
    let gather_dims = engine.buf_uniform(GatherDims { rows, embed, _p0: 0, _p1: 0 }, "gather_dims");
    let gather_bg = engine.bind_group(
        &engine.gather_add,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: model.type_embeddings.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: idx_buf.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: gather_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(&mut encoder, &engine.gather_add, &gather_bg, ((rows * embed).div_ceil(256), 1, 1), "gather_add");

    let time_mask_buf = engine.buf_f32(&time_mask_flat, "time_mask");
    let group_mask_buf = engine.buf_f32(&group_mask_flat, "group_mask");
    let cos_buf = engine.buf_f32(&rope_tables.cos, "rope_cos");
    let sin_buf = engine.buf_f32(&rope_tables.sin, "rope_sin");
    let scale_buf = engine.buf_f32(&rope_tables.scale, "rope_scale");
    let scale_inv_buf = engine.buf_f32(&rope_tables.scale_inv, "rope_scale_inv");

    for layer in &model.layers {
        let normed = rmsnorm_full(engine, &mut encoder, &x, &layer.norm_scale, rows, embed);

        let attn_out = match layer.ty {
            LayerType::Time => {
                let qkv = linear(engine, &mut encoder, &normed, rows, embed, &layer.w_qkv, &layer.b_qkv, 3 * embed, 0);
                rmsnorm_qk(engine, &mut encoder, &qkv, &layer.q_norm, rows, heads, 0, 3 * embed);
                rmsnorm_qk(engine, &mut encoder, &qkv, &layer.k_norm, rows, heads, embed, 3 * embed);
                rope(engine, &mut encoder, &qkv, &cos_buf, &sin_buf, &scale_buf, rows, heads, 0, 3 * embed, p);
                rope(engine, &mut encoder, &qkv, &cos_buf, &sin_buf, &scale_inv_buf, rows, heads, embed, 3 * embed, p);
                let attn_pre = attention(engine, &mut encoder, &qkv, &time_mask_buf, v, p, heads, embed);
                linear(engine, &mut encoder, &attn_pre, rows, embed, &layer.w_o, &layer.b_o, embed, 0)
            }
            LayerType::Group => {
                let normed_t = transpose_outer(engine, &mut encoder, &normed, v, p, embed);
                let qkv = linear(engine, &mut encoder, &normed_t, rows, embed, &layer.w_qkv, &layer.b_qkv, 3 * embed, 0);
                rmsnorm_qk(engine, &mut encoder, &qkv, &layer.q_norm, rows, heads, 0, 3 * embed);
                rmsnorm_qk(engine, &mut encoder, &qkv, &layer.k_norm, rows, heads, embed, 3 * embed);
                let attn_pre = attention(engine, &mut encoder, &qkv, &group_mask_buf, p, v, heads, embed);
                let attn_out_t = linear(engine, &mut encoder, &attn_pre, rows, embed, &layer.w_o, &layer.b_o, embed, 0);
                transpose_outer(engine, &mut encoder, &attn_out_t, p, v, embed)
            }
        };
        add_inplace(engine, &mut encoder, &x, &attn_out, rows * embed);

        let mlp_normed = rmsnorm_full(engine, &mut encoder, &x, &layer.mlp_norm_scale, rows, embed);
        let w0_out = linear(engine, &mut encoder, &mlp_normed, rows, embed, &layer.w0, &layer.b0, 2 * layer.mlp_hidden, 0);
        let gated = silu_mul(engine, &mut encoder, &w0_out, rows, layer.mlp_hidden);
        let mlp_out = linear(engine, &mut encoder, &gated, rows, layer.mlp_hidden, &layer.w2, &layer.b2, embed, 0);
        add_inplace(engine, &mut encoder, &x, &mlp_out, rows * embed);
    }

    let normed_final = rmsnorm_full(engine, &mut encoder, &x, &model.out_norm_scale, rows, embed);
    let decoded = residual_block(engine, &mut encoder, &normed_final, rows, &model.decoder); // [rows, patch_size*n_q]

    let n_q = cfg.n_quantiles() as u32;
    let quantile_rows = rows * patch_size;
    let q_dims = engine.buf_uniform(QuantileDims { rows: quantile_rows, n_q, _p0: 0, _p1: 0 }, "quantile_dims");
    let out_buf = engine.buf_empty((quantile_rows * n_q) as usize, "quantile_out");
    let q_bg = engine.bind_group(
        &engine.quantile_head,
        &[
            BindGroupEntry { binding: 0, resource: decoded.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: q_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(&mut encoder, &engine.quantile_head, &q_bg, (quantile_rows.div_ceil(64), 1, 1), "quantile_head");

    engine.queue.submit(Some(encoder.finish()));
    Ok(engine.read_buffer(&out_buf, (quantile_rows * n_q) as usize).await)
}
