//! Batch path: N independent signals in one forward pass must equal N
//! separate single-signal forward passes, because the only isolation
//! mechanism is per-signal group ids feeding the existing group-attention
//! mask (see `TimeSeries::from_context_batch`) — no new attention code.
//! `#[ignore]`d: needs the t0-alpha checkpoint (never committed).

use std::path::PathBuf;

use burn_ndarray::NdArray;
use serde::Deserialize;
use t0_core::{forecast, forecast_batch, T0Config, T0Model, Weights};

type B = NdArray<f32>;

#[derive(Deserialize)]
struct Manifest {
    context_len: usize,
    horizon: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    v: usize,
    context_file: String,
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
#[ignore = "requires the t0-alpha checkpoint; run `hf download theforecastingcompany/t0-alpha --local-dir ~/Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha` first, then `cargo test -- --ignored`"]
fn batch_of_8_copies_equals_8x_single() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap()).unwrap();
    let case = &manifest.cases[0];
    assert_eq!(case.v, 1, "fixture 0 must be univariate for this batch test");

    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha");
    assert!(model_dir.join("model.safetensors").exists(), "{} not found — see the #[ignore] reason above", model_dir.display());

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::load(&model_dir.join("model.safetensors")).unwrap();
    let device = Default::default();
    let model = T0Model::<B>::load(&weights, config, &device).unwrap();

    let context = read_f32(&root.join(&case.context_file));
    let (single, _) = forecast(&model, &context, 1, manifest.context_len, manifest.horizon, &device, false);

    let n = 8;
    let batched_context: Vec<f32> = context.iter().cloned().cycle().take(n * context.len()).collect();
    let batched = forecast_batch(&model, &batched_context, n, manifest.context_len, manifest.horizon, &device);

    assert_eq!(batched.len(), n * single.len());
    for i in 0..n {
        let slice = &batched[i * single.len()..(i + 1) * single.len()];
        let max_abs = slice.iter().zip(&single).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "batch copy {i} diverges from single-signal forecast by {max_abs:.3e}");
    }
}
