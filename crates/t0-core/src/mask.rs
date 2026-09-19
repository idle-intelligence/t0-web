//! Attention masks, mirroring `t0.mask` (`tfc-t0` PyPI package,
//! `t0/mask.py`). Computed on the CPU from per-timestep mask/group-id/
//! variate-type arrays (small: at most a few hundred patches × variates for
//! this milestone) and uploaded as additive float masks (`0.0` = attend,
//! a large negative = blocked) so the attention kernel only needs one add.

use crate::data::{MaskType, VariateType};

const BLOCK: f32 = -1.0e9;

/// `t0.mask.reduce_patch_metadata`: each patch's first non-`PAD` timestep's
/// value, or `-1` if the whole patch is `PAD`.
pub fn reduce_patch_metadata(patched_meta: &[i64], patched_mask: &[i8], v: usize, patch_size: usize) -> Vec<i64> {
    let t = patched_meta.len() / v;
    let p = t / patch_size;
    let mut out = vec![-1i64; v * p];
    for row in 0..v {
        for patch in 0..p {
            let base = row * t + patch * patch_size;
            for k in 0..patch_size {
                if patched_mask[base + k] != MaskType::Pad as i8 {
                    out[row * p + patch] = patched_meta[base + k];
                    break;
                }
            }
        }
    }
    out
}

/// `t0.mask.compute_patch_attention_mask`: `true` unless every timestep in
/// the patch is `PAD`.
pub fn patch_attendable(patched_mask: &[i8], v: usize, patch_size: usize) -> Vec<bool> {
    let t = patched_mask.len() / v;
    let p = t / patch_size;
    let mut out = vec![false; v * p];
    for row in 0..v {
        for patch in 0..p {
            let base = row * t + patch * patch_size;
            out[row * p + patch] = (0..patch_size).any(|k| patched_mask[base + k] != MaskType::Pad as i8);
        }
    }
    out
}

/// `t0.mask.MaskBuilder.build_time_mask`, flattened `[v, p, p]` (the head
/// broadcast dim is added by the caller via `unsqueeze`). Causal for
/// `TARGET`/`HISTORICAL` rows, bidirectional for `FUTURE` rows, restricted
/// to same-group and non-padding keys.
pub fn build_time_mask(patch_group_ids: &[i64], patch_variate_type: &[i64], attendable: &[bool], v: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v * p * p];
    for row in 0..v {
        let is_future = patch_variate_type[row * p] == VariateType::Future as i64;
        for i in 0..p {
            for j in 0..p {
                let gi = patch_group_ids[row * p + i];
                let gj = patch_group_ids[row * p + j];
                let same_doc = gi >= 0 && gi == gj;
                let causal_ok = is_future || j <= i;
                let attend = same_doc && causal_ok && attendable[row * p + j];
                out[row * p * p + i * p + j] = if attend { 0.0 } else { BLOCK };
            }
        }
    }
    out
}

/// `t0.mask.MaskBuilder.build_group_mask`, flattened `[p, v, v]`. Variates
/// attend to others sharing the same (non-padding) group id at that patch.
pub fn build_group_mask(patch_group_ids: &[i64], v: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; p * v * v];
    for patch in 0..p {
        for i in 0..v {
            for j in 0..v {
                let gi = patch_group_ids[i * p + patch];
                let gj = patch_group_ids[j * p + patch];
                let attend = gi >= 0 && gi == gj;
                out[patch * v * v + i * v + j] = if attend { 0.0 } else { BLOCK };
            }
        }
    }
    out
}
