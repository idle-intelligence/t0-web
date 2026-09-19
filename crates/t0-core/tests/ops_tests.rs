//! Tiny hand-computed cases for the primitives most likely to have a sign
//! or convention bug (RoPE's interleaved-pair rotation cost us one already —
//! see `docs/BENCHMARKS.md`). QK-norm reuses `rmsnorm` verbatim, so the
//! RMSNorm case covers it too.

use burn_ndarray::NdArray;
use t0_core::ops::{rmsnorm, RopeTables};

type B = NdArray<f32>;

#[test]
fn rmsnorm_matches_hand_computation() {
    let device = Default::default();
    // x = [3, 4]; mean(x^2) = 12.5; denom = sqrt(12.5 + 1e-8) ~= 3.5355339.
    let x: burn::tensor::Tensor<B, 2> = burn::tensor::Tensor::from_floats([[3.0f32, 4.0]], &device);
    let scale: burn::tensor::Tensor<B, 1> = burn::tensor::Tensor::from_floats([1.0f32, 1.0], &device);
    let out = rmsnorm(x, scale, 1e-8);
    let got: Vec<f32> = out.into_data().iter::<f32>().collect();
    let denom = 12.5f64.sqrt();
    let expected = [3.0 / denom as f32, 4.0 / denom as f32];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
    }
}

#[test]
fn rmsnorm_applies_scale() {
    let device = Default::default();
    let x: burn::tensor::Tensor<B, 2> = burn::tensor::Tensor::from_floats([[1.0f32, 1.0, 1.0, 1.0]], &device);
    // mean(x^2) = 1, denom = 1, so output == scale exactly.
    let scale: burn::tensor::Tensor<B, 1> = burn::tensor::Tensor::from_floats([2.0f32, 0.5, -1.0, 3.0], &device);
    let out = rmsnorm(x, scale, 1e-8);
    let got: Vec<f32> = out.into_data().iter::<f32>().collect();
    assert_eq!(got, vec![2.0, 0.5, -1.0, 3.0]);
}

#[test]
fn rope_zero_position_is_identity_rotation() {
    // At position 0, angle = 0 for every frequency, so cos=1, sin=0 and the
    // xpos scale power is (0 - max_half)/scale_base -- only an identity
    // rotation when max_half is also 0 (seq_len 1 or 2), which this checks.
    let device = Default::default();
    let tables = RopeTables::<B>::new(1, 4, &device);
    let cos: Vec<f32> = tables.cos.into_data().iter::<f32>().collect();
    let sin: Vec<f32> = tables.sin.into_data().iter::<f32>().collect();
    let scale: Vec<f32> = tables.scale.into_data().iter::<f32>().collect();
    assert_eq!(cos, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(sin, vec![0.0, 0.0, 0.0, 0.0]);
    for s in scale {
        assert!((s - 1.0).abs() < 1e-6, "xpos scale at seq_len=1 should be base^0=1, got {s}");
    }
}

#[test]
fn rope_scale_table_is_halves_concat_not_interleaved() {
    // Regression test for the bug this port hit: `rotary_embedding_torch`'s
    // xpos `scale` duplicates via `cat([s, s])` (mirrored halves), while
    // `cos`/`sin` duplicate via interleaving (`f0,f0,f1,f1,...`) to match
    // `rotate_half`'s adjacent-pair convention. Getting this wrong cost
    // ~5e-3 layer-0 max-abs error against the reference (see
    // docs/BENCHMARKS.md, milestone-0 parity run).
    let device = Default::default();
    let head_dim = 4;
    let tables = RopeTables::<B>::new(3, head_dim, &device); // seq_len > 1 so scale != 1 everywhere
    let scale: Vec<f32> = tables.scale.into_data().iter::<f32>().collect();
    // Row for position p=2 (index 2*head_dim..3*head_dim): halves-concat
    // means scale[0] == scale[2] and scale[1] == scale[3] (mirrored halves),
    // NOT scale[0] == scale[1] (which interleaving would give).
    let row = &scale[2 * head_dim..3 * head_dim];
    assert!((row[0] - row[2]).abs() < 1e-9, "halves-concat: dim 0 and dim (half+0) must match");
    assert!((row[1] - row[3]).abs() < 1e-9, "halves-concat: dim 1 and dim (half+1) must match");
}
