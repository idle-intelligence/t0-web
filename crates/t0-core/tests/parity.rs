//! Numerical parity against the reference PyTorch fixtures in `fixtures/`
//! (see `tools/make_fixtures.py`). `#[ignore]`d with a reason: it needs both
//! the fixtures (committed, small) and the t0-alpha checkpoint (not
//! committed — weights are never committed).
//!
//! `t0-cli parity` runs the same check with per-fixture output for manual
//! divergence-hunting; this test is the CI-shaped "did it stay passing" gate.

use std::path::PathBuf;

use burn_ndarray::NdArray;
use serde::Deserialize;
use t0_core::{forecast, T0Config, T0Model, Weights};

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
    quantiles_file: String,
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
fn quantiles_match_reference_within_1e_minus_4() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap()).unwrap();

    let home = std::env::var("HOME").unwrap();
    let model_dir = PathBuf::from(home).join("Code/idle-intelligence/models/hf/theforecastingcompany/t0-alpha");
    assert!(model_dir.join("model.safetensors").exists(), "{} not found — see the #[ignore] reason above", model_dir.display());

    let config: T0Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::load(&model_dir.join("model.safetensors")).unwrap();
    let device = Default::default();
    let model = T0Model::<B>::load(&weights, config, &device).unwrap();

    for case in &manifest.cases {
        let context = read_f32(&root.join(&case.context_file));
        let expected = read_f32(&root.join(&case.quantiles_file));
        let (got, _) = forecast(&model, &context, case.v, manifest.context_len, manifest.horizon, &device, false);
        assert_eq!(got.len(), expected.len());
        let max_abs = got.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-4, "max-abs error {max_abs:.3e} exceeds 1e-4 gate");
    }
}
