//! Q8_0/Q4_0 GPU residency for the four big per-layer matmuls
//! (`wQKV`, `wO`, `mlp.0`, `mlp.2` -- same set `t0_core::weights::is_quantizable`
//! names), with in-kernel dequant (`shaders/linear_q8.wgsl`,
//! `shaders/linear_q4.wgsl`). Block math matches `t0_core::gguf` exactly
//! (`quantize_q8_0`/`quantize_q4_0`); this module only repacks those same
//! blocks into a GPU-friendly layout: the block's f16 scale decoded to f32
//! once host-side into its own buffer (so the kernel never needs to decode
//! f16), and the quantized values packed 4/u32 so WGSL can read them as
//! `array<u32>` without a byte-addressed storage buffer.

use t0_core::gguf::{dequantize_for, quantize_q4_0, quantize_q8_0, GgmlType};

use crate::engine::Engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuant {
    F32,
    Q8_0,
    Q4_0,
}

impl WeightQuant {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "f32" => WeightQuant::F32,
            "q8_0" => WeightQuant::Q8_0,
            "q4_0" => WeightQuant::Q4_0,
            other => anyhow::bail!("unknown t0-fast weight quant {other}, expected f32|q8_0|q4_0"),
        })
    }
}

pub enum MatMulWeight {
    F32 { w: wgpu::Buffer },
    Q8_0 { qs: wgpu::Buffer, scales: wgpu::Buffer, blocks_per_row: u32 },
    Q4_0 { qs: wgpu::Buffer, scales: wgpu::Buffer, blocks_per_row: u32 },
}

impl MatMulWeight {
    /// GPU bytes actually resident for this weight (for the memory
    /// before/after report) -- `out_dim * in_dim * 4` for F32, or
    /// `qs.size() + scales.size()` for the quantized variants.
    pub fn gpu_bytes(&self) -> u64 {
        match self {
            MatMulWeight::F32 { w } => w.size(),
            MatMulWeight::Q8_0 { qs, scales, .. } | MatMulWeight::Q4_0 { qs, scales, .. } => qs.size() + scales.size(),
        }
    }
}

/// `QK=32` per llama.cpp / `t0_core::gguf`. Block bytes: Q8_0 = 2 (f16) + 32
/// (i8); Q4_0 = 2 (f16) + 16 (nibbles).
const QK: usize = 32;

fn split_q8_blocks(bytes: &[u8], n_elements: usize) -> (Vec<u32>, Vec<f32>) {
    let n_blocks = n_elements / QK;
    let mut qs = vec![0u32; n_elements / 4];
    let mut scales = vec![0f32; n_blocks];
    for (bi, block) in bytes.chunks_exact(2 + QK).enumerate() {
        scales[bi] = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..QK {
            let byte = block[2 + j] as u32;
            let word_idx = bi * (QK / 4) + j / 4;
            let shift = (j % 4) * 8;
            qs[word_idx] |= byte << shift;
        }
    }
    (qs, scales)
}

fn split_q4_blocks(bytes: &[u8], n_elements: usize) -> (Vec<u32>, Vec<f32>) {
    let n_blocks = n_elements / QK;
    let mut qs = vec![0u32; n_blocks * 4]; // 16 bytes/block = 4 u32/block
    let mut scales = vec![0f32; n_blocks];
    for (bi, block) in bytes.chunks_exact(2 + QK / 2).enumerate() {
        scales[bi] = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for byte_i in 0..QK / 2 {
            let byte = block[2 + byte_i] as u32;
            let word_idx = bi * 4 + byte_i / 4;
            let shift = (byte_i % 4) * 8;
            qs[word_idx] |= byte << shift;
        }
    }
    (qs, scales)
}

/// Same as `load_matmul_weight`, but for a tensor read straight out of a
/// GGUF file (`ty`/`bytes` from `t0_core::gguf::read_gguf`'s `GgufFile::tensors`):
/// if it's already Q8_0/Q4_0 on disk, its block bytes go straight into
/// `split_q8_blocks`/`split_q4_blocks` -- no dequantize-then-requantize
/// round trip, and no F32-expanded copy of the weight ever exists in
/// memory (the two-phase-loading point of this function: the GGUF reader's
/// `Vec<u8>` for this tensor can be dropped right after this call). F32/F16
/// on disk (small tensors, or a `--quant f16` export) still dequantizes to
/// f32 -- there's no lower-precision GPU compute path for those today.
pub fn load_matmul_weight_gguf(engine: &Engine, label: &str, shape: &[usize], ty: GgmlType, bytes: &[u8]) -> MatMulWeight {
    let in_dim = shape[1];
    match ty {
        GgmlType::Q8_0 => {
            let n_elements: usize = shape.iter().product();
            let (qs, scales) = split_q8_blocks(bytes, n_elements);
            MatMulWeight::Q8_0 {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (in_dim / QK) as u32,
            }
        }
        GgmlType::Q4_0 => {
            let n_elements: usize = shape.iter().product();
            let (qs, scales) = split_q4_blocks(bytes, n_elements);
            MatMulWeight::Q4_0 {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (in_dim / QK) as u32,
            }
        }
        GgmlType::F32 | GgmlType::F16 => {
            let n_elements: usize = shape.iter().product();
            let data = dequantize_for(ty, bytes, n_elements);
            MatMulWeight::F32 { w: engine.buf_f32(&data, label) }
        }
    }
}

/// Loads one big matmul weight (`shape = [out_dim, in_dim]`, PyTorch
/// layout) at the given residency. `label` is a load-time buffer label
/// only (not a `Pool` key -- these buffers live for the model's lifetime,
/// not per-forward).
pub fn load_matmul_weight(engine: &Engine, label: &str, shape: &[usize], data: &[f32], quant: WeightQuant) -> MatMulWeight {
    let in_dim = shape[1];
    match quant {
        WeightQuant::F32 => MatMulWeight::F32 { w: engine.buf_f32(data, label) },
        WeightQuant::Q8_0 => {
            assert_eq!(in_dim % QK, 0, "{label}: in_dim {in_dim} not a multiple of {QK}");
            let bytes = quantize_q8_0(data);
            let (qs, scales) = split_q8_blocks(&bytes, data.len());
            MatMulWeight::Q8_0 {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (in_dim / QK) as u32,
            }
        }
        WeightQuant::Q4_0 => {
            assert_eq!(in_dim % QK, 0, "{label}: in_dim {in_dim} not a multiple of {QK}");
            let bytes = quantize_q4_0(data);
            let (qs, scales) = split_q4_blocks(&bytes, data.len());
            MatMulWeight::Q4_0 {
                qs: engine.buf_u32(&qs, &format!("{label}.qs")),
                scales: engine.buf_f32(&scales, &format!("{label}.scales")),
                blocks_per_row: (in_dim / QK) as u32,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use t0_core::gguf::dequantize_q8_0;

    #[test]
    fn q8_roundtrip_matches_reference_dequant() {
        let n = 64usize;
        let mut data = vec![0f32; n];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f32) - 32.0) * 0.37;
        }
        let bytes = quantize_q8_0(&data);
        let deq = dequantize_q8_0(&bytes, n);
        let (qs, scales) = split_q8_blocks(&bytes, n);

        let mut out = vec![0f32; n];
        for (col, o) in out.iter_mut().enumerate() {
            let blk = col / 32;
            let j = col % 32;
            let word_idx = blk * 8 + j / 4;
            let shift = (j % 4) * 8;
            let word = qs[word_idx];
            let byteval = (word >> shift) & 0xFF;
            let sv = ((byteval as i32) << 24) >> 24;
            *o = sv as f32 * scales[blk];
        }
        let maxdiff = deq.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert_eq!(maxdiff, 0.0, "deq={deq:?} out={out:?}");
    }
}
