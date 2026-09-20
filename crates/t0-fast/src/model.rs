//! GPU-resident `T0Model` forward pass with no Burn tensor anywhere: every
//! op in `t0_core::ops` (rmsnorm, linear, RoPE, mhsa, swiglu_ffn,
//! quantile_head) has a hand-written WGSL counterpart in `engine.rs`'s
//! pipeline set, and `forward_async` records every dispatch for one
//! forecast into a single `wgpu::CommandEncoder`, submits once, and reads
//! back once (`Engine::read_buffer`). CPU-side prep (patch concat, masks,
//! the causal scaler, RoPE table math) is reused as-is from `t0_core` --
//! none of that touches a GPU buffer until the single upload per input.

use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::BindGroupEntry;

use t0_core::config::LayerType;
use t0_core::data::{MaskType, TimeSeries};
use t0_core::mask::{build_group_mask, build_time_mask, patch_attendable, reduce_patch_metadata};
use t0_core::{T0Config, Weights};

use crate::engine::Engine;
use crate::pool::Pool;
use crate::quant::{load_matmul_weight, load_matmul_weight_gguf, MatMulWeight, WeightQuant};
use crate::rope_tables::RopeTables;

const TILED_THRESHOLD_ROWS: u32 = 32;

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
struct LinearQDims {
    m: u32,
    k: u32,
    n: u32,
    act: u32,
    blocks_per_row: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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
    w_qkv: MatMulWeight,
    b_qkv: wgpu::Buffer,
    w_o: MatMulWeight,
    b_o: wgpu::Buffer,
    q_norm: wgpu::Buffer,
    k_norm: wgpu::Buffer,
    mlp_norm_scale: wgpu::Buffer,
    w0: MatMulWeight,
    b0: wgpu::Buffer,
    w2: MatMulWeight,
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
    /// This model's own working-buffer/bind-group pool -- never shared with
    /// another `GpuModel`, even one built from the same `Engine` (same
    /// device/queue/pipelines). Pool cache keys are bare call-site strings
    /// with no per-model or per-weight-quant-type component (see
    /// `pool.rs`'s module doc), so two models sharing one `Pool` would
    /// silently reuse each other's bind groups -- which are bound to the
    /// *previous* model's weight buffers -- whenever both hit the same
    /// `(v, p)` shape. See docs/runs/2026-09-20-compare-page.md for the
    /// bug this was.
    pub pool: Pool,
}

impl GpuModel {
    /// Sum of GPU-resident bytes for the four big per-layer matmuls across
    /// all layers (the tensors `quant::load_matmul_weight` can quantize) --
    /// for the Phase B memory before/after report. Excludes the patch
    /// encoder/decoder/norms/type-embeddings, which always stay F32.
    pub fn quantizable_gpu_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.w_qkv.gpu_bytes() + l.w_o.gpu_bytes() + l.w0.gpu_bytes() + l.w2.gpu_bytes())
            .sum()
    }

    fn residual_block_bytes(w: &ResidualBlockBuf) -> u64 {
        w.hidden_w.size() + w.hidden_b.size() + w.output_w.size() + w.output_b.size() + w.residual_w.size() + w.residual_b.size()
    }

    /// Every persistent GPU buffer this model holds (weights only -- not
    /// the per-forward `Pool` working set, see `GpuModel::pool`'s
    /// `resident_bytes`). The GPU-memory report a browser page can show.
    pub fn total_weight_bytes(&self) -> u64 {
        let mut total = Self::residual_block_bytes(&self.patch_encoder) + self.type_embeddings.size() + self.out_norm_scale.size() + Self::residual_block_bytes(&self.decoder);
        for l in &self.layers {
            total += l.norm_scale.size()
                + l.b_qkv.size()
                + l.b_o.size()
                + l.q_norm.size()
                + l.k_norm.size()
                + l.mlp_norm_scale.size()
                + l.b0.size()
                + l.b2.size()
                + l.w_qkv.gpu_bytes()
                + l.w_o.gpu_bytes()
                + l.w0.gpu_bytes()
                + l.w2.gpu_bytes();
        }
        total
    }
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

type GgufTensors = std::collections::HashMap<String, (Vec<usize>, t0_core::gguf::GgmlType, Vec<u8>)>;

fn gguf_get<'a>(tensors: &'a GgufTensors, name: &str) -> Result<&'a (Vec<usize>, t0_core::gguf::GgmlType, Vec<u8>)> {
    tensors.get(name).ok_or_else(|| anyhow!("gguf: missing tensor {name}"))
}

fn gguf_f32(engine: &Engine, tensors: &GgufTensors, name: &str) -> Result<wgpu::Buffer> {
    let (shape, ty, bytes) = gguf_get(tensors, name)?;
    let n: usize = shape.iter().product();
    let data = t0_core::gguf::dequantize_for(*ty, bytes, n);
    Ok(engine.buf_f32(&data, name))
}

/// Same as `load_residual_block`, but for tensors read straight out of a
/// GGUF file: the patch encoder/decoder are never quantized (see
/// `t0_core::weights::is_quantizable`), so this always dequantizes to f32,
/// same as the small per-layer tensors in `load_from_gguf_bytes` below.
fn load_residual_block_gguf(engine: &Engine, tensors: &GgufTensors, prefix: &str) -> Result<ResidualBlockBuf> {
    let (hw_shape, _, _) = gguf_get(tensors, &format!("{prefix}.mlp.hidden_layer.weight"))?;
    let in_dim = hw_shape[1] as u32;
    let hidden_dim = hw_shape[0] as u32;
    let (ow_shape, _, _) = gguf_get(tensors, &format!("{prefix}.mlp.output_layer.weight"))?;
    let out_dim = ow_shape[0] as u32;
    Ok(ResidualBlockBuf {
        hidden_w: gguf_f32(engine, tensors, &format!("{prefix}.mlp.hidden_layer.weight"))?,
        hidden_b: gguf_f32(engine, tensors, &format!("{prefix}.mlp.hidden_layer.bias"))?,
        output_w: gguf_f32(engine, tensors, &format!("{prefix}.mlp.output_layer.weight"))?,
        output_b: gguf_f32(engine, tensors, &format!("{prefix}.mlp.output_layer.bias"))?,
        residual_w: gguf_f32(engine, tensors, &format!("{prefix}.residual_layer.weight"))?,
        residual_b: gguf_f32(engine, tensors, &format!("{prefix}.residual_layer.bias"))?,
        in_dim,
        hidden_dim,
        out_dim,
    })
}

impl GpuModel {
    /// Loads a `t0-cli export-gguf` file with the big per-layer matmuls
    /// (wQKV/wO/mlp.0/mlp.2) kept resident at whatever quantization the
    /// GGUF already has (Q8_0/Q4_0) -- no dequantize-then-requantize round
    /// trip, and no F32-expanded copy of those tensors is ever built (the
    /// two-phase-loading point: `t0_core::gguf::read_gguf`'s `Vec<u8>` per
    /// tensor is consumed tensor-by-tensor and dropped as this function
    /// returns, never held alongside the GPU-resident copy). Everything
    /// else (patch encoder, decoder, norms, type embeddings) is F16 on
    /// disk regardless of `--quant` and is dequantized to f32 here, same
    /// as `Weights::load_gguf_bytes` -- those tensors are tiny and always
    /// F32-resident on the GPU (see `load`'s doc comment / `docs/reports/
    /// t0-alpha.md` §5 on why: they gate output precision, not worth
    /// quantizing).
    pub fn load_from_gguf_bytes(engine: &Engine, gguf_bytes: &[u8]) -> Result<Self> {
        let file = t0_core::gguf::read_gguf(&mut &gguf_bytes[..]).context("parsing gguf")?;
        let config = t0_core::weights::config_from_gguf_metadata(&file.metadata)?;
        let tensors = &file.tensors;

        let patch_encoder = load_residual_block_gguf(engine, tensors, "patch_encoder.projection")?;
        let type_embeddings = gguf_f32(engine, tensors, "patch_encoder.type_embeddings.weight")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, ty) in config.layer_types().into_iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            let (w_qkv_shape, w_qkv_ty, w_qkv_bytes) = gguf_get(tensors, &format!("{p}.attention_block.attention.wQKV.weight"))?;
            let (w_o_shape, w_o_ty, w_o_bytes) = gguf_get(tensors, &format!("{p}.attention_block.attention.wO.weight"))?;
            let (w0_shape, w0_ty, w0_bytes) = gguf_get(tensors, &format!("{p}.mlp.0.weight"))?;
            let (w2_shape, w2_ty, w2_bytes) = gguf_get(tensors, &format!("{p}.mlp.2.weight"))?;
            let mlp_hidden = (w0_shape[0] / 2) as u32;
            layers.push(LayerBuf {
                ty,
                norm_scale: gguf_f32(engine, tensors, &format!("{p}.attention_block.norm.scale"))?,
                w_qkv: load_matmul_weight_gguf(engine, "w_qkv", w_qkv_shape, *w_qkv_ty, w_qkv_bytes),
                b_qkv: gguf_f32(engine, tensors, &format!("{p}.attention_block.attention.wQKV.bias"))?,
                w_o: load_matmul_weight_gguf(engine, "w_o", w_o_shape, *w_o_ty, w_o_bytes),
                b_o: gguf_f32(engine, tensors, &format!("{p}.attention_block.attention.wO.bias"))?,
                q_norm: gguf_f32(engine, tensors, &format!("{p}.attention_block.attention.q_norm.scale"))?,
                k_norm: gguf_f32(engine, tensors, &format!("{p}.attention_block.attention.k_norm.scale"))?,
                mlp_norm_scale: gguf_f32(engine, tensors, &format!("{p}.norm.scale"))?,
                w0: load_matmul_weight_gguf(engine, "w0", w0_shape, *w0_ty, w0_bytes),
                b0: gguf_f32(engine, tensors, &format!("{p}.mlp.0.bias"))?,
                w2: load_matmul_weight_gguf(engine, "w2", w2_shape, *w2_ty, w2_bytes),
                b2: gguf_f32(engine, tensors, &format!("{p}.mlp.2.bias"))?,
                mlp_hidden,
            });
        }

        let out_norm_scale = gguf_f32(engine, tensors, "transformer.out_norm.scale")?;
        let decoder = load_residual_block_gguf(engine, tensors, "decoder")?;

        Ok(GpuModel {
            config,
            patch_encoder,
            type_embeddings,
            layers,
            out_norm_scale,
            decoder,
            pool: Pool::new(engine.device.clone(), engine.queue.clone()),
        })
    }

    pub fn load(engine: &Engine, weights: &Weights, config: T0Config, quant: WeightQuant) -> Result<Self> {
        let patch_encoder = load_residual_block(engine, weights, "patch_encoder.projection")?;
        let (_, type_emb) = weights.get_raw("patch_encoder.type_embeddings.weight")?;
        let type_embeddings = engine.buf_f32(type_emb, "type_embeddings");

        let mut layers = Vec::with_capacity(config.num_layers);
        for (i, ty) in config.layer_types().into_iter().enumerate() {
            let p = format!("transformer.layers.{i}");
            let (_, norm_scale) = weights.get_raw(&format!("{p}.attention_block.norm.scale"))?;
            let (w_qkv_shape, w_qkv) = weights.get_raw(&format!("{p}.attention_block.attention.wQKV.weight"))?;
            let (_, b_qkv) = weights.get_raw(&format!("{p}.attention_block.attention.wQKV.bias"))?;
            let (w_o_shape, w_o) = weights.get_raw(&format!("{p}.attention_block.attention.wO.weight"))?;
            let (_, b_o) = weights.get_raw(&format!("{p}.attention_block.attention.wO.bias"))?;
            let (_, q_norm) = weights.get_raw(&format!("{p}.attention_block.attention.q_norm.scale"))?;
            let (_, k_norm) = weights.get_raw(&format!("{p}.attention_block.attention.k_norm.scale"))?;
            let (_, mlp_norm_scale) = weights.get_raw(&format!("{p}.norm.scale"))?;
            let (w0_shape, w0) = weights.get_raw(&format!("{p}.mlp.0.weight"))?;
            let (_, b0) = weights.get_raw(&format!("{p}.mlp.0.bias"))?;
            let (w2_shape, w2) = weights.get_raw(&format!("{p}.mlp.2.weight"))?;
            let (_, b2) = weights.get_raw(&format!("{p}.mlp.2.bias"))?;
            let mlp_hidden = (w0_shape[0] / 2) as u32;
            layers.push(LayerBuf {
                ty,
                norm_scale: engine.buf_f32(norm_scale, "norm_scale"),
                w_qkv: load_matmul_weight(engine, "w_qkv", w_qkv_shape, w_qkv, quant),
                b_qkv: engine.buf_f32(b_qkv, "b_qkv"),
                w_o: load_matmul_weight(engine, "w_o", w_o_shape, w_o, quant),
                b_o: engine.buf_f32(b_o, "b_o"),
                q_norm: engine.buf_f32(q_norm, "q_norm"),
                k_norm: engine.buf_f32(k_norm, "k_norm"),
                mlp_norm_scale: engine.buf_f32(mlp_norm_scale, "mlp_norm_scale"),
                w0: load_matmul_weight(engine, "w0", w0_shape, w0, quant),
                b0: engine.buf_f32(b0, "b0"),
                w2: load_matmul_weight(engine, "w2", w2_shape, w2, quant),
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
            pool: Pool::new(engine.device.clone(), engine.queue.clone()),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn linear(
    engine: &Engine,
    pool: &Pool,
    encoder: &mut wgpu::CommandEncoder,
    key: &str,
    x: &wgpu::Buffer,
    rows: u32,
    in_dim: u32,
    w: &MatMulWeight,
    b: &wgpu::Buffer,
    out_dim: u32,
    act: u32,
) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * out_dim) as usize);
    // Naive kernels are one thread per output element on a 16x16 workgroup
    // (div_ceil(16) each axis); the tiled f32/Q8_0 kernels use 2x2 register
    // blocking, one workgroup (still 16x16=256 threads) covering a 32x32
    // output block (div_ceil(32) each axis) -- see linear_tiled.wgsl.
    let wgs_naive = (out_dim.div_ceil(16), rows.div_ceil(16), 1);
    let wgs_tiled32 = (out_dim.div_ceil(32), rows.div_ceil(32), 1);
    match w {
        MatMulWeight::F32 { w } => {
            // Naive is one thread per output element with zero reuse across
            // rows -- fine at M=1 (single-forecast decode), wasteful once M
            // grows (batching): the tiled kernel amortizes both x and w
            // reads across a 32x32 block (2x2 register-blocked) instead of
            // re-reading them per thread. See docs/runs/2026-09-20-perf.md's
            // Phase C tiled-GEMM entry for the native/browser batch-24
            // numbers this threshold is based on.
            let tiled = rows >= TILED_THRESHOLD_ROWS;
            let pipeline = if tiled { &engine.linear_tiled } else { &engine.linear };
            let wgs = if tiled { wgs_tiled32 } else { wgs_naive };
            let dims = pool.uniform(&format!("{key}.dims"), LinearDims { m: rows, k: in_dim, n: out_dim, act });
            // Bind-group cache key must include `tiled`: naive/tiled are
            // different pipelines with distinct auto-derived (`layout:
            // None`) bind-group layouts, even though their WGSL bindings
            // look identical -- reusing a bind group built against one
            // pipeline's layout on the other's pass invalidates the
            // command encoder (surfaces as "Encoder is invalid" at
            // `finish()`, not at the bad `set_bind_group` call). Calls
            // whose `rows` varies across invocations at the same call site
            // (e.g. `gifteval`'s differently-sized windows through one
            // resident model) cross `TILED_THRESHOLD_ROWS` in both
            // directions, so the plain `key` alone aliased two pipelines.
            let bg_key = format!("{key}.{}", if tiled { "tiled" } else { "naive" });
            let bg = pool.bind_group(
                &bg_key,
                pipeline,
                &[
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
                ],
            );
            engine.dispatch(encoder, pipeline, &bg, wgs, key);
        }
        MatMulWeight::Q8_0 { qs, scales, blocks_per_row } | MatMulWeight::Q4_0 { qs, scales, blocks_per_row } => {
            // With the earlier 16x16/1-output-per-thread tiled kernel,
            // Q4_0's cheap dequant meant the naive kernel already won
            // (batch-24 native: 36.2ms/signal naive vs 41.8ms/signal
            // tiled). After adding 2x2 register blocking (each thread
            // reuses its dequantized b_tile registers across 2 output
            // columns instead of 1), Q4_0 tiled now wins too (34.5 ->
            // 22.2ms/signal, see docs/runs/2026-09-20-perf.md) -- so all
            // three weight residencies route through the tiled kernel once
            // `rows >= TILED_THRESHOLD_ROWS`.
            let tiled = rows >= TILED_THRESHOLD_ROWS;
            let pipeline = match (matches!(w, MatMulWeight::Q8_0 { .. }), tiled) {
                (true, true) => &engine.linear_tiled_q8,
                (true, false) => &engine.linear_q8,
                (false, true) => &engine.linear_tiled_q4,
                (false, false) => &engine.linear_q4,
            };
            let wgs = if tiled { wgs_tiled32 } else { wgs_naive };
            let dims = pool.uniform(
                &format!("{key}.dims"),
                LinearQDims { m: rows, k: in_dim, n: out_dim, act, blocks_per_row: *blocks_per_row, _p0: 0, _p1: 0, _p2: 0 },
            );
            // See the F32 branch above: bind-group cache key must include `tiled`.
            let bg_key = format!("{key}.{}", if tiled { "tiled" } else { "naive" });
            let bg = pool.bind_group(
                &bg_key,
                pipeline,
                &[
                    BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: qs.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: scales.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: b.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: dims.as_entire_binding() },
                ],
            );
            engine.dispatch(encoder, pipeline, &bg, wgs, key);
        }
    }
    out
}

fn add_inplace(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) {
    let dims = pool.uniform(&format!("{key}.dims"), AddDims { len, _p0: 0, _p1: 0, _p2: 0 });
    let bg = pool.bind_group(
        key,
        &engine.add_inplace,
        &[
            BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.add_inplace, &bg, (len.div_ceil(256), 1, 1), key);
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm_full(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, scale: &wgpu::Buffer, rows: u32, dim: u32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * dim) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), RmsFullDims { rows, dim, _p0: 0, _p1: 0 });
    let bg = pool.bind_group(
        key,
        &engine.rmsnorm_full,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm_full, &bg, (rows.div_ceil(64), 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn rmsnorm_qk(
    engine: &Engine,
    pool: &Pool,
    encoder: &mut wgpu::CommandEncoder,
    key: &str,
    buf: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    rows: u32,
    heads: u32,
    head_dim: u32,
    base_offset: u32,
    row_stride: u32,
) {
    let dims = pool.uniform(
        &format!("{key}.dims"),
        RmsQkDims { rows, heads, head_dim, base_offset, row_stride, _p0: 0, _p1: 0, _p2: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.rmsnorm_qk,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.rmsnorm_qk, &bg, ((rows * heads).div_ceil(64), 1, 1), key);
}

#[allow(clippy::too_many_arguments)]
fn rope(
    engine: &Engine,
    pool: &Pool,
    encoder: &mut wgpu::CommandEncoder,
    key: &str,
    buf: &wgpu::Buffer,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    rows: u32,
    heads: u32,
    head_dim: u32,
    base_offset: u32,
    row_stride: u32,
    seq_len: u32,
) {
    let dims = pool.uniform(
        &format!("{key}.dims"),
        RopeDims { rows, heads, head_dim, base_offset, row_stride, seq_len, _p0: 0, _p1: 0 },
    );
    let bg = pool.bind_group(
        key,
        &engine.rope,
        &[
            BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: cos.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: sin.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: scale.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
        ],
    );
    let half = head_dim / 2;
    engine.dispatch(encoder, &engine.rope, &bg, ((rows * heads * half).div_ceil(64), 1, 1), key);
}

/// `seq` (patch count for time layers, variate-chunk size for group layers)
/// is capped by `MAX_SEQ` in `attention.wgsl` (256, WebGPU's workgroup-
/// invocation cap -- one thread per seq position -- and enough for
/// `forecast_rollout`'s long-context windows, e.g. 160 patches at
/// context=4096/horizon=1024). One kernel, one code path regardless of
/// `seq`; this assert only catches a config this crate has never been
/// exercised against.
#[allow(clippy::too_many_arguments)]
fn attention(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, qkv: &wgpu::Buffer, mask: &wgpu::Buffer, outer: u32, seq: u32, heads: u32, embed: u32) -> wgpu::Buffer {
    debug_assert!(seq <= 256, "seq_len {seq} exceeds MAX_SEQ=256 in attention.wgsl");
    let head_dim = embed / heads;
    let qkv_stride = 3 * embed;
    let out = pool.data(&format!("{key}.out"), (outer * seq * embed) as usize);
    let dims = pool.uniform(
        &format!("{key}.dims"),
        AttnDims {
            outer,
            seq,
            heads,
            embed,
            qkv_stride,
            mask_outer_stride: seq * seq,
            scale: 1.0 / (head_dim as f32).sqrt(),
            _p0: 0,
        },
    );
    let pipeline = engine.attention_pipeline(head_dim);
    let bg = pool.bind_group(
        key,
        &pipeline,
        &[
            BindGroupEntry { binding: 0, resource: qkv.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: mask.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &pipeline, &bg, (outer * heads, 1, 1), key);
    out
}

#[allow(clippy::too_many_arguments)]
fn transpose_outer(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, src: &wgpu::Buffer, a: u32, b: u32, e: u32) -> wgpu::Buffer {
    let dst = pool.data(&format!("{key}.out"), (a * b * e) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), TransposeDims { a, b, e, _p0: 0 });
    let bg = pool.bind_group(
        key,
        &engine.transpose_outer,
        &[
            BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.transpose_outer, &bg, ((a * b).div_ceil(64), 1, 1), key);
    dst
}

fn silu_mul(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, src: &wgpu::Buffer, rows: u32, hidden: u32) -> wgpu::Buffer {
    let out = pool.data(&format!("{key}.out"), (rows * hidden) as usize);
    let dims = pool.uniform(&format!("{key}.dims"), SiluDims { rows, hidden, _p0: 0, _p1: 0 });
    let bg = pool.bind_group(
        key,
        &engine.silu_mul,
        &[
            BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
        ],
    );
    engine.dispatch(encoder, &engine.silu_mul, &bg, ((rows * hidden).div_ceil(256), 1, 1), key);
    out
}

fn residual_block(engine: &Engine, pool: &Pool, encoder: &mut wgpu::CommandEncoder, key: &str, x: &wgpu::Buffer, rows: u32, w: &ResidualBlockBuf) -> wgpu::Buffer {
    let hidden_w = MatMulWeight::F32 { w: w.hidden_w.clone() };
    let output_w = MatMulWeight::F32 { w: w.output_w.clone() };
    let residual_w = MatMulWeight::F32 { w: w.residual_w.clone() };
    let hidden = linear(engine, pool, encoder, &format!("{key}.hidden"), x, rows, w.in_dim, &hidden_w, &w.hidden_b, w.hidden_dim, 1);
    let out = linear(engine, pool, encoder, &format!("{key}.out"), &hidden, rows, w.hidden_dim, &output_w, &w.output_b, w.out_dim, 0);
    let residual = linear(engine, pool, encoder, &format!("{key}.residual"), x, rows, w.in_dim, &residual_w, &w.residual_b, w.out_dim, 0);
    add_inplace(engine, pool, encoder, &format!("{key}.add"), &out, &residual, rows * w.out_dim);
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

    let head_dim = embed / heads;

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
    let rope_tables = RopeTables::new(p as usize, head_dim as usize);

    let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t0-fast forward") });
    let pool = &model.pool;

    let concat_buf = pool.upload_f32("concat", &concat);
    let x = residual_block(engine, pool, &mut encoder, "patch_enc", &concat_buf, rows, &model.patch_encoder);

    let idx_buf = pool.upload_u32("type_idx", &type_idx);
    let gather_dims = pool.uniform("gather.dims", GatherDims { rows, embed, _p0: 0, _p1: 0 });
    let gather_bg = pool.bind_group(
        "gather",
        &engine.gather_add,
        &[
            BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: model.type_embeddings.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: idx_buf.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: gather_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(&mut encoder, &engine.gather_add, &gather_bg, ((rows * embed).div_ceil(256), 1, 1), "gather");

    let time_mask_buf = pool.upload_f32("time_mask", &time_mask_flat);
    let group_mask_buf = pool.upload_f32("group_mask", &group_mask_flat);
    let cos_buf = pool.upload_f32("rope_cos", &rope_tables.cos);
    let sin_buf = pool.upload_f32("rope_sin", &rope_tables.sin);
    let scale_buf = pool.upload_f32("rope_scale", &rope_tables.scale);
    let scale_inv_buf = pool.upload_f32("rope_scale_inv", &rope_tables.scale_inv);

    for (i, layer) in model.layers.iter().enumerate() {
        let lk = |s: &str| format!("layer{i}.{s}");
        let normed = rmsnorm_full(engine, pool, &mut encoder, &lk("norm"), &x, &layer.norm_scale, rows, embed);

        let attn_out = match layer.ty {
            LayerType::Time => {
                let qkv = linear(engine, pool, &mut encoder, &lk("qkv"), &normed, rows, embed, &layer.w_qkv, &layer.b_qkv, 3 * embed, 0);
                rmsnorm_qk(engine, pool, &mut encoder, &lk("qnorm"), &qkv, &layer.q_norm, rows, heads, head_dim, 0, 3 * embed);
                rmsnorm_qk(engine, pool, &mut encoder, &lk("knorm"), &qkv, &layer.k_norm, rows, heads, head_dim, embed, 3 * embed);
                rope(engine, pool, &mut encoder, &lk("ropeq"), &qkv, &cos_buf, &sin_buf, &scale_buf, rows, heads, head_dim, 0, 3 * embed, p);
                rope(engine, pool, &mut encoder, &lk("ropek"), &qkv, &cos_buf, &sin_buf, &scale_inv_buf, rows, heads, head_dim, embed, 3 * embed, p);
                let attn_pre = attention(engine, pool, &mut encoder, &lk("attn"), &qkv, &time_mask_buf, v, p, heads, embed);
                linear(engine, pool, &mut encoder, &lk("wo"), &attn_pre, rows, embed, &layer.w_o, &layer.b_o, embed, 0)
            }
            LayerType::Group => {
                let normed_t = transpose_outer(engine, pool, &mut encoder, &lk("tr1"), &normed, v, p, embed);
                let qkv = linear(engine, pool, &mut encoder, &lk("qkv"), &normed_t, rows, embed, &layer.w_qkv, &layer.b_qkv, 3 * embed, 0);
                rmsnorm_qk(engine, pool, &mut encoder, &lk("qnorm"), &qkv, &layer.q_norm, rows, heads, head_dim, 0, 3 * embed);
                rmsnorm_qk(engine, pool, &mut encoder, &lk("knorm"), &qkv, &layer.k_norm, rows, heads, head_dim, embed, 3 * embed);
                let attn_pre = attention(engine, pool, &mut encoder, &lk("attn"), &qkv, &group_mask_buf, p, v, heads, embed);
                let attn_out_t = linear(engine, pool, &mut encoder, &lk("wo"), &attn_pre, rows, embed, &layer.w_o, &layer.b_o, embed, 0);
                transpose_outer(engine, pool, &mut encoder, &lk("tr2"), &attn_out_t, p, v, embed)
            }
        };
        add_inplace(engine, pool, &mut encoder, &lk("add1"), &x, &attn_out, rows * embed);

        let mlp_normed = rmsnorm_full(engine, pool, &mut encoder, &lk("mlpnorm"), &x, &layer.mlp_norm_scale, rows, embed);
        let w0_out = linear(engine, pool, &mut encoder, &lk("w0"), &mlp_normed, rows, embed, &layer.w0, &layer.b0, 2 * layer.mlp_hidden, 0);
        let gated = silu_mul(engine, pool, &mut encoder, &lk("silu"), &w0_out, rows, layer.mlp_hidden);
        let mlp_out = linear(engine, pool, &mut encoder, &lk("w2"), &gated, rows, layer.mlp_hidden, &layer.w2, &layer.b2, embed, 0);
        add_inplace(engine, pool, &mut encoder, &lk("add2"), &x, &mlp_out, rows * embed);
    }

    let normed_final = rmsnorm_full(engine, pool, &mut encoder, "out_norm", &x, &model.out_norm_scale, rows, embed);
    let decoded = residual_block(engine, pool, &mut encoder, "decoder", &normed_final, rows, &model.decoder); // [rows, patch_size*n_q]

    let n_q = cfg.n_quantiles() as u32;
    let quantile_rows = rows * patch_size;
    let q_dims = pool.uniform("quantile.dims", QuantileDims { rows: quantile_rows, n_q, _p0: 0, _p1: 0 });
    let out_buf = pool.data("quantile.out", (quantile_rows * n_q) as usize);
    let q_bg = pool.bind_group(
        "quantile",
        &engine.quantile_head,
        &[
            BindGroupEntry { binding: 0, resource: decoded.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: q_dims.as_entire_binding() },
        ],
    );
    engine.dispatch(&mut encoder, &engine.quantile_head, &q_bg, (quantile_rows.div_ceil(64), 1, 1), "quantile");

    engine.queue.submit(Some(encoder.finish()));
    Ok(engine.read_buffer(&out_buf, (quantile_rows * n_q) as usize).await)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{load_matmul_weight, WeightQuant};

    /// Runs `linear()` at `rows=m` against a direct dequant-and-dot
    /// reference (`t0_core::gguf::dequantize_{q8_0,q4_0}` for the
    /// quantized cases, the raw f32 weight for `WeightQuant::F32`) and
    /// asserts float32-rounding agreement. `m` selects naive
    /// (`< TILED_THRESHOLD_ROWS`) vs the tiled shared-memory kernel
    /// (`>=`), so calling this at both `m=1` and `m=40` (with the same
    /// `k=512, n=4` weight) guards both kernel families for all three
    /// weight residencies without duplicating the setup three times.
    fn check_linear_kernel(quant: WeightQuant, m: usize) {
        let engine = Engine::new().unwrap();
        let k = 512usize;
        let n = 4usize;
        let mut w = vec![0f32; n * k];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5) * 0.6;
        }
        let bias = vec![0.0f32; n];
        let x: Vec<f32> = (0..m * k).map(|i| (((i * 40503) % 1000) as f32 / 1000.0 - 0.5) * 0.4).collect();

        let dequant = match quant {
            WeightQuant::F32 => w.clone(),
            WeightQuant::Q8_0 => {
                let bytes = t0_core::gguf::quantize_q8_0(&w);
                t0_core::gguf::dequantize_q8_0(&bytes, w.len())
            }
            WeightQuant::Q4_0 => {
                let bytes = t0_core::gguf::quantize_q4_0(&w);
                t0_core::gguf::dequantize_q4_0(&bytes, w.len())
            }
        };
        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            for (col_n, e) in expected[row * n..(row + 1) * n].iter_mut().enumerate() {
                *e = (0..k).map(|col_k| x[row * k + col_k] * dequant[col_n * k + col_k]).sum::<f32>() + bias[col_n];
            }
        }

        let mm = load_matmul_weight(&engine, "test_w", &[n, k], &w, quant);
        let pool = Pool::new(engine.device.clone(), engine.queue.clone());
        let x_buf = pool.upload_f32("test_x", &x);
        let b_buf = pool.upload_f32("test_b", &bias);
        let mut encoder = engine.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let out = linear(&engine, &pool, &mut encoder, "test_linear", &x_buf, m as u32, k as u32, &mm, &b_buf, n as u32, 0);
        engine.queue.submit(Some(encoder.finish()));
        let got = pollster::block_on(engine.read_buffer(&out, m * n));

        for i in 0..m * n {
            assert!((got[i] - expected[i]).abs() < 1e-5, "quant={quant:?} m={m} i={i}: got {} expected {}", got[i], expected[i]);
        }
    }

    #[test]
    fn f32_linear_naive() {
        check_linear_kernel(WeightQuant::F32, 1);
    }
    #[test]
    fn f32_linear_tiled() {
        check_linear_kernel(WeightQuant::F32, 40);
    }
    #[test]
    fn q8_linear_naive() {
        check_linear_kernel(WeightQuant::Q8_0, 1);
    }
    #[test]
    fn q8_linear_tiled() {
        check_linear_kernel(WeightQuant::Q8_0, 40);
    }
    #[test]
    fn q4_linear_naive() {
        check_linear_kernel(WeightQuant::Q4_0, 1);
    }
    #[test]
    fn q4_linear_tiled() {
        check_linear_kernel(WeightQuant::Q4_0, 40);
    }
}
