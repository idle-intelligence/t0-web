//! Tensor-level building blocks, generic over the Burn `Backend`. Ported
//! from `tfc-t0`'s `t0/model/layers/{norm,rope,time_attention,
//! group_attention,mlp,feed_forward,head}.py` — see each function's doc
//! comment for the exact reference it matches.

use burn::tensor::activation::{relu, silu, softmax, softplus};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// `t0.model.layers.norm.RMSNorm`: `x / sqrt(mean(x^2, -1) + eps) * scale`,
/// no bias. `eps = 1e-8` everywhere in the reference.
pub fn rmsnorm<B: Backend, const D: usize>(x: Tensor<B, D>, scale: Tensor<B, 1>, eps: f64) -> Tensor<B, D> {
    let ms = x.clone().powf_scalar(2.0f32).mean_dim(D - 1);
    let denom = (ms + eps).sqrt();
    let normed = x / denom;
    normed * scale.unsqueeze::<D>()
}

/// `t0.model.layers.mlp.MLP` + `.ResidualBlock`: `mlp(x) + residual_layer(x)`,
/// where `mlp` is `Linear -> activation(ReLU) -> Linear` (dropout is
/// inference-time inert). Used by the patch encoder's `projection` and by
/// `decoder`. `x` is `[rows, input]`, weights are PyTorch `[out, in]`.
#[allow(clippy::too_many_arguments)]
pub fn residual_block<B: Backend>(
    x: Tensor<B, 2>,
    hidden_w: Tensor<B, 2>,
    hidden_b: Tensor<B, 1>,
    output_w: Tensor<B, 2>,
    output_b: Tensor<B, 1>,
    residual_w: Tensor<B, 2>,
    residual_b: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let hidden = relu(linear(x.clone(), hidden_w, hidden_b));
    let out = linear(hidden, output_w, output_b);
    let residual = linear(x, residual_w, residual_b);
    out + residual
}

/// PyTorch `nn.Linear`: `x @ w.T + b`, `w` is `[out, in]`.
pub fn linear<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>, b: Tensor<B, 1>) -> Tensor<B, 2> {
    x.matmul(w.transpose()) + b.unsqueeze()
}

/// `t0.model.layers.feed_forward.SwiGLU` composed with the two
/// `nn.Linear`s around it (`TransformerLayer.mlp`): `down(silu(gate) * up)`
/// where `[gate, up] = chunk(hidden_w(x), 2, dim=-1)` — gate is the *first*
/// half of the fused `2*mlp_hidden_dim`-wide projection, `up` the second
/// (xFormers ordering, per the reference's comment).
pub fn swiglu_ffn<B: Backend>(
    x: Tensor<B, 2>,
    w0: Tensor<B, 2>,
    b0: Tensor<B, 1>,
    w2: Tensor<B, 2>,
    b2: Tensor<B, 1>,
    mlp_hidden_dim: usize,
) -> Tensor<B, 2> {
    let hidden = linear(x, w0, b0); // [rows, 2*mlp_hidden_dim]
    let gate = hidden.clone().narrow(1, 0, mlp_hidden_dim);
    let up = hidden.narrow(1, mlp_hidden_dim, mlp_hidden_dim);
    let gated = silu(gate) * up;
    linear(gated, w2, b2)
}

/// Precomputed RoPE tables for one patch-sequence length, broadcastable to
/// `[outer, heads, seq, head_dim]` via `.unsqueeze::<4>()` at the call site.
/// `t0.model.layers.rope.TimeAwareRotaryEmbedding` (adjacent-pair /
/// "interleaved" convention from the `rotary_embedding_torch` dependency —
/// *not* the split-half convention some other decoders use), `theta=10000`,
/// `use_xpos=True`, `xpos_scale_base=512` (library default, unoverridden).
#[derive(Clone)]
pub struct RopeTables<B: Backend> {
    pub cos: Tensor<B, 2>,   // [seq, head_dim]
    pub sin: Tensor<B, 2>,   // [seq, head_dim]
    pub scale: Tensor<B, 2>, // [seq, head_dim], applied to q; k uses its reciprocal
}

impl<B: Backend> RopeTables<B> {
    pub fn new(seq_len: usize, head_dim: usize, device: &B::Device) -> Self {
        assert_eq!(head_dim % 2, 0);
        let half = head_dim / 2;
        let theta = 10000f64;
        let scale_base = 512f64;
        let max_pos = (seq_len - 1) as f64;
        let max_half = (max_pos / 2.0).floor();

        let mut cos = vec![0f32; seq_len * head_dim];
        let mut sin = vec![0f32; seq_len * head_dim];
        let mut scale = vec![0f32; seq_len * head_dim];
        for p in 0..seq_len {
            let power = (p as f64 - max_half) / scale_base;
            for j in 0..half {
                let inv_freq = 1.0 / theta.powf((2 * j) as f64 / head_dim as f64);
                let angle = p as f64 * inv_freq;
                let base = (2.0 * j as f64 + 0.4 * head_dim as f64) / (1.4 * head_dim as f64);
                let s = base.powf(power) as f32;
                // cos/sin: interleaved-pair duplication (repeat(r=2): f0,f0,f1,f1,...),
                // matching `rotate_half`'s adjacent-pair convention.
                for k in 0..2 {
                    let idx = p * head_dim + 2 * j + k;
                    cos[idx] = angle.cos() as f32;
                    sin[idx] = angle.sin() as f32;
                }
                // xpos scale: halves-concat duplication (`cat([s, s], dim=-1)`),
                // *not* interleaved like cos/sin — a real asymmetry in the
                // upstream `rotary_embedding_torch` library, reproduced exactly.
                scale[p * head_dim + j] = s;
                scale[p * head_dim + half + j] = s;
            }
        }
        RopeTables {
            cos: Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([seq_len, head_dim]),
            sin: Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([seq_len, head_dim]),
            scale: Tensor::<B, 1>::from_floats(scale.as_slice(), device).reshape([seq_len, head_dim]),
        }
    }
}

/// Adjacent-pair rotation: `x` is `[outer, heads, seq, head_dim]`; pairs
/// `(x[2j], x[2j+1])` rotate to `(-x[2j+1], x[2j])` — `rotary_embedding_torch`'s
/// `rotate_half` (despite the name, an interleaved-pair convention).
fn rotate_half_interleaved<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [o, h, s, d] = x.dims();
    let x5 = x.reshape([o, h, s, d / 2, 2]);
    let x1 = x5.clone().narrow(4, 0, 1);
    let x2 = x5.narrow(4, 1, 1);
    Tensor::cat(vec![-x2, x1], 4).reshape([o, h, s, d])
}

/// Apply RoPE to `x` (`[outer, heads, seq, head_dim]`); `invert_scale`
/// selects q's (`false`) or k's (`true`, `scale**-1`) xpos scaling, per
/// `TimeAwareRotaryEmbedding.rotate_queries_and_keys`.
pub fn apply_rope<B: Backend>(x: Tensor<B, 4>, tables: &RopeTables<B>, invert_scale: bool) -> Tensor<B, 4> {
    let cos = tables.cos.clone().unsqueeze::<4>();
    let sin = tables.sin.clone().unsqueeze::<4>();
    let scale = if invert_scale {
        tables.scale.clone().recip()
    } else {
        tables.scale.clone()
    };
    let scale = scale.unsqueeze::<4>();
    x.clone() * cos * scale.clone() + rotate_half_interleaved(x) * sin * scale
}

/// Multi-head self-attention shared by time- and group-attention blocks:
/// QK-RMSNorm (per head, before RoPE), scaled dot-product attention with an
/// additive mask, `wO` output projection. `x` is `[outer, seq, embed]`
/// (`outer` is the batch axis attention does *not* run over — variates for
/// time-attention, patches for group-attention); `additive_mask` is
/// `[outer_or_1, 1, seq, seq]`, `0.0` where attend, a large negative where
/// blocked. `rope` is `None` for group-attention (`t0.model.layers
/// .group_attention.VariateSelfAttention` explicitly skips it).
#[allow(clippy::too_many_arguments)]
pub fn mhsa<B: Backend>(
    x: Tensor<B, 3>,
    w_qkv: Tensor<B, 2>,
    b_qkv: Tensor<B, 1>,
    w_o: Tensor<B, 2>,
    b_o: Tensor<B, 1>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
    num_heads: usize,
    additive_mask: Tensor<B, 4>,
    rope: Option<&RopeTables<B>>,
) -> Tensor<B, 3> {
    let [outer, seq, embed] = x.dims();
    let head_dim = embed / num_heads;

    let qkv = linear(x.reshape([outer * seq, embed]), w_qkv, b_qkv);
    let qkv = qkv.reshape([outer, seq, 3, num_heads, head_dim]);
    let qkv = qkv.permute([2usize, 0, 3, 1, 4]); // [3, outer, heads, seq, head_dim]
    let q = qkv.clone().narrow(0, 0, 1).reshape([outer, num_heads, seq, head_dim]);
    let k = qkv.clone().narrow(0, 1, 1).reshape([outer, num_heads, seq, head_dim]);
    let v = qkv.narrow(0, 2, 1).reshape([outer, num_heads, seq, head_dim]);

    let eps = 1e-8;
    let mut q = rmsnorm(q, q_norm, eps);
    let mut k = rmsnorm(k, k_norm, eps);
    if let Some(tables) = rope {
        q = apply_rope(q, tables, false);
        k = apply_rope(k, tables, true);
    }

    let scores = q.matmul(k.swap_dims(2, 3)) * (1.0 / (head_dim as f32).sqrt());
    let scores = scores + additive_mask;
    let attn = softmax(scores, 3);
    let out = attn.matmul(v); // [outer, heads, seq, head_dim]
    let out = out.swap_dims(1, 2).reshape([outer, seq, embed]);

    linear(out.reshape([outer * seq, embed]), w_o, b_o).reshape([outer, seq, embed])
}

/// `t0.model.layers.head.QuantileHead` (monotonicity via cumsum-of-softplus):
/// `x` is `[rows, n_quantiles]` sorted ascending;
/// `q_0 = raw_0`, `q_i = q_0 + sum_{j<=i} softplus(raw_j)` for `i > 0`.
pub fn quantile_head<B: Backend>(x: Tensor<B, 2>, n_quantiles: usize) -> Tensor<B, 2> {
    if n_quantiles == 1 {
        return x;
    }
    let first = x.clone().narrow(1, 0, 1);
    let rest = x.narrow(1, 1, n_quantiles - 1);
    let increments = softplus(rest, 1.0);
    let mut cols = vec![first.clone()];
    let mut running = first;
    for i in 0..n_quantiles - 1 {
        let inc = increments.clone().narrow(1, i, 1);
        running = running + inc;
        cols.push(running.clone());
    }
    Tensor::cat(cols, 1)
}
