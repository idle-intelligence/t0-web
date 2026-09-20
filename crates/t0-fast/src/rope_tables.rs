//! Plain-`Vec<f32>` port of `t0_core::ops::RopeTables::new` (interleaved-pair
//! xpos RoPE, `theta=10000`, `xpos_scale_base=512`) -- identical math, no
//! Burn tensor. `scale_inv` is the k-side table (`ops.rs` calls
//! `tables.scale.recip()` at the `apply_rope` call site instead of building
//! a second table; we bake the reciprocal in host-side once instead, since
//! there's no Burn `Tensor::recip` to call here).

pub struct RopeTables {
    pub cos: Vec<f32>,       // [seq_len, head_dim]
    pub sin: Vec<f32>,       // [seq_len, head_dim]
    pub scale: Vec<f32>,     // [seq_len, head_dim], applied to q
    pub scale_inv: Vec<f32>, // [seq_len, head_dim], applied to k
}

impl RopeTables {
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
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
                for k in 0..2 {
                    let idx = p * head_dim + 2 * j + k;
                    cos[idx] = angle.cos() as f32;
                    sin[idx] = angle.sin() as f32;
                }
                scale[p * head_dim + j] = s;
                scale[p * head_dim + half + j] = s;
            }
        }
        let scale_inv: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        RopeTables { cos, sin, scale, scale_inv }
    }
}
